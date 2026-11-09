"""Read an explicitly inventoried HVAC evaluation bundle without filling its data.

This module validates source evidence. Loading a bundle does not qualify its
observations, fit a model or authorize planning with its contents.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Self
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .types import ModeSpan, utc


class BundleArtifact(BaseModel):
    """An immutable file identity recorded by the bundle exporter."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["parquet", "json"]


class BundleManifest(BaseModel):
    """Versioned inventory and provenance, separate from training eligibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    bundle_version: Literal[1] = 1
    exported_at: AwareDatetime
    start: AwareDatetime
    end: AwareDatetime
    controller_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    timezone: str
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    ems_commit: str = Field(min_length=7)
    ingest_lag_min: int | None = Field(default=None, ge=0)
    training_ready: bool = False
    files: dict[str, BundleArtifact]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @model_validator(mode="after")
    def _ordered(self) -> BundleManifest:
        if self.end <= self.start:
            raise ValueError("bundle end must follow start")
        if self.training_ready and self.ingest_lag_min is None:
            raise ValueError("training-ready bundle requires measured ingestion lag")
        return self


def _utc_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f"{name}: timestamps must be timezone-aware")
    if (
        str(index.tz) != "UTC"
        or not index.is_monotonic_increasing
        or not index.is_unique
    ):
        raise ValueError(f"{name}: timestamps must be sorted, unique UTC")
    return index


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _validate_minutes(frame: pd.DataFrame, manifest: BundleManifest) -> None:
    index = _utc_index(frame.index, "minutes")
    if "power_w" not in frame or not index.floor("min").equals(index):
        raise ValueError("minutes require power_w at minute starts")
    if len(index) and (
        (index < pd.Timestamp(manifest.start)).any()
        or (index >= pd.Timestamp(manifest.end)).any()
    ):
        raise ValueError("minutes fall outside bundle span")
    values = pd.to_numeric(frame.power_w, errors="raise")
    if values.isin([float("inf"), -float("inf")]).any() or (values < 0).any():
        raise ValueError("invalid electrical power")


@dataclass(frozen=True)
class HvacEvaluationBundle:
    """Validated private-source inventory usable by the evaluation API.

    Tables preserve observed gaps, provenance and extra quality columns. Numeric
    model code is deliberately not imported by the loader.
    """

    manifest: BundleManifest
    tables: Mapping[str, pd.DataFrame]
    documents: Mapping[str, JsonValue]

    @classmethod
    def load(cls, directory: Path) -> HvacEvaluationBundle:
        """Read only files listed in a manifest and verify their exact identities.

        Args:
            directory: Export directory containing manifest.json and its files.

        Returns:
            Validated tables/documents with missing observations preserved.

        Raises:
            ValueError: If inventory, hashes or the minute contract are invalid.
            OSError: If an inventoried file cannot be read.
        """
        root = directory.resolve(strict=True)
        manifest = BundleManifest.model_validate(
            json.loads(
                (root / "manifest.json").read_bytes(), parse_constant=_reject_constant
            )
        )
        tables: dict[str, pd.DataFrame] = {}
        documents: dict[str, JsonValue] = {}
        for name, spec in manifest.files.items():
            if Path(name).name != name or name in {".", "..", "manifest.json"}:
                raise ValueError("artifact name must be a local filename")
            path = (root / name).resolve(strict=True)
            if path.parent != root:
                raise ValueError("artifact escapes bundle directory")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != spec.sha256:
                raise ValueError(f"artifact hash mismatch: {name}")
            if spec.format == "parquet":
                tables[name] = pd.read_parquet(io.BytesIO(data))
            else:
                documents[name] = json.loads(data, parse_constant=_reject_constant)
        if "minutes.parquet" not in tables or "capabilities.json" not in documents:
            raise ValueError("bundle requires minutes and capability evidence")
        if not isinstance(documents["capabilities.json"], dict):
            raise ValueError("capability evidence must be a JSON object")
        _validate_minutes(tables["minutes.parquet"], manifest)
        return cls(manifest, MappingProxyType(tables), MappingProxyType(documents))

    def available_at(self, origin: datetime) -> pd.DataFrame:
        """Return an isolated observed-history slice after the availability boundary.

        Args:
            origin: Aware UTC origin of a fit or prediction.

        Returns:
            Copy of minute rows whose end plus measured lag is at/before origin.

        Raises:
            ValueError: If the origin is not UTC or ingestion lag is unmeasured.
        """
        offset = origin.utcoffset()
        if origin.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("origin must be aware UTC")
        if self.manifest.ingest_lag_min is None:
            raise ValueError("ingestion lag is not measured")
        frame = self.tables["minutes.parquet"]
        cutoff = pd.Timestamp(origin) - pd.Timedelta(
            minutes=1 + self.manifest.ingest_lag_min
        )
        return frame.loc[frame.index <= cutoff].copy(deep=True)


# Compatibility mirror of the owning EMS policy. Synthetic golden cases are
# generated by the actual common/state_engine functions; see hvac_schedule_parity.
# This is an archive adapter, not an independent configurable scheduling policy.

_SCHEDULE_OFF: Literal["OFF"] = "OFF"
_AUTOMATION_OPEN_ENDED_HOLD = timedelta(hours=2)
_SCHEDULE_VALIDITY = timedelta(minutes=15)
_MAX_SCHEDULE_HORIZON = timedelta(days=4)
_FUTURE_SKEW = timedelta(seconds=30)
_ScheduleMode = Literal["OFF", "ECO", "NORMAL", "BOOST"]


class _ScheduleSlot(BaseModel):
    """Validated archived EMS slot, retaining an unknown mode."""

    start: AwareDatetime
    end: AwareDatetime
    mode: _ScheduleMode | None
    source: Literal["plan", "automation"] = "plan"
    reason: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.end <= self.start:
            raise ValueError("slot end must follow start")
        return self


class _ScheduleOverride(BaseModel):
    """Archived automation intent with its original expiry."""

    mode: _ScheduleMode
    source: Literal["automation"] = "automation"
    valid_until: AwareDatetime | None = None
    reason: str | None = Field(default=None, max_length=200)


class _DeviceSchedule(BaseModel):
    """Bounded contiguous device schedule from the EMS wire contract."""

    device_id: str = Field(min_length=1, max_length=128)
    slots: list[_ScheduleSlot] = Field(default_factory=list, max_length=384)
    override: _ScheduleOverride | None = None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if any(a.end != b.start for a, b in pairwise(self.slots)):
            raise ValueError("schedule slots must be ordered and contiguous")
        if (
            self.slots
            and self.slots[-1].end - self.slots[0].start > _MAX_SCHEDULE_HORIZON
        ):
            raise ValueError("schedule exceeds maximum horizon")
        return self


class _SchedulePayload(BaseModel):
    """EMS schedule-version mirror for isolated offline reconstruction."""

    schema_version: Literal[1] = 1
    generated_at: AwareDatetime
    valid_until: AwareDatetime
    plan_generated_at: AwareDatetime | None = None
    thermostat_enabled: bool = False
    comfort_off_latched: bool = False
    devices: list[_DeviceSchedule] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (
            not 0
            < (self.valid_until - self.generated_at).total_seconds()
            <= _SCHEDULE_VALIDITY.total_seconds()
        ):
            raise ValueError("invalid schedule validity")
        ids = [d.device_id for d in self.devices]
        if len(ids) != len(set(ids)):
            raise ValueError("device ids must be unique")
        return self

    def is_current(self, now: datetime) -> bool:
        return self.generated_at <= now + _FUTURE_SKEW and now < self.valid_until

    def device(self, device_id: str) -> _DeviceSchedule | None:
        return next((d for d in self.devices if d.device_id == device_id), None)


def _project_automation_override(
    grid: list[_ScheduleSlot], override: _ScheduleOverride | None, *, now: datetime
) -> list[_ScheduleSlot]:
    if override is None:
        return list(grid)
    until = override.valid_until or now + _AUTOMATION_OPEN_ENDED_HOLD
    return [
        (
            slot.model_copy(
                update={
                    "mode": override.mode,
                    "source": "automation",
                    "reason": override.reason,
                }
            )
            if slot.start < until
            else slot
        )
        for slot in grid
    ]


def _schedule_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("HVAC projection timestamps must be timezone-aware")
    return utc(value)


def effective_mode_spans(
    payload: Mapping[str, JsonValue] | None,
    device_id: str,
    origin: datetime,
    last_mode: str | None,
    last_mode_event: datetime | None,
    *,
    horizon_end: datetime | None = None,
) -> tuple[ModeSpan, ...]:
    """Apply the fixture-verified EMS S2 policy to one archived schedule version.

    Args:
        payload: Full JSON version known at the origin, or None.
        device_id: Selected heat pump.
        origin: Causal UTC prediction origin.
        last_mode: Actual command held outside known spans.
        last_mode_event: Causal latching comfort-OFF timestamp, when active.
        horizon_end: Requested grid end.

    Returns:
        Sorted nonoverlapping known/uncertain spans with automation precedence.

    Raises:
        ValueError: For malformed archives or an active latch with unknown age.
    """
    parsed = _SchedulePayload.model_validate(payload) if payload is not None else None
    return _project_effective_mode_spans(
        parsed, device_id, origin, last_mode, last_mode_event, horizon_end=horizon_end
    )


def _project_effective_mode_spans(
    payload: _SchedulePayload | None,
    device_id: str,
    origin: datetime,
    last_mode: str | None,
    last_mode_event: datetime | None,
    *,
    horizon_end: datetime | None = None,
) -> tuple[ModeSpan, ...]:
    """Project plan, automation and an already-active comfort OFF latch.

    Args:
        payload: Schedule available at the origin; stale/missing means no spans.
        device_id: One heat pump's identity.
        origin: Actual forecast origin, not the upcoming output-grid boundary.
        last_mode: Last actual command, held by kpower in uncovered periods.
        last_mode_event: Timestamp of the latching thermostat OFF event when the
            comfort latch is active. It must be causal; no duration is invented.
        horizon_end: Optional requested grid end. The default covers the plan and
            the 24-hour upper bound on an active latch.

    Returns:
        Sorted nonoverlapping spans. None plan slots remain uncovered. Automation
        has precedence over comfort. Split comfort spans preserve underlying modes
        and elapsed latch age, permitting one shared sampled release per path.

    Raises:
        ValueError: On invalid timestamps or an active latch without its event age.
    """
    origin = _schedule_utc(origin)
    if payload is None or not payload.is_current(origin):
        return ()
    device = payload.device(device_id)
    if device is None:
        return ()
    comfort = payload.thermostat_enabled and payload.comfort_off_latched
    latch_at = _schedule_utc(last_mode_event) if last_mode_event is not None else None
    if comfort and (latch_at is None or latch_at > origin):
        raise ValueError("active comfort latch requires its causal OFF event")
    end = (
        _schedule_utc(horizon_end)
        if horizon_end is not None
        else max(
            [
                origin + timedelta(hours=24),
                *(_schedule_utc(slot.end) for slot in device.slots),
            ]
        )
    )
    if end <= origin:
        raise ValueError("projection horizon must follow origin")
    comfort_end = min(origin + timedelta(hours=24), end)
    boundaries = {origin, end}
    if comfort:
        boundaries.add(comfort_end)
    for slot in device.slots:
        boundaries.update(
            at
            for at in (_schedule_utc(slot.start), _schedule_utc(slot.end))
            if origin < at < end
        )
    if device.override is not None:
        until = (
            _schedule_utc(device.override.valid_until)
            if device.override.valid_until
            else origin + _AUTOMATION_OPEN_ENDED_HOLD
        )
        if origin < until < end:
            boundaries.add(until)
    grid = []
    underlying: list[str | None] = []
    for start, stop in pairwise(sorted(boundaries)):
        covering = next(
            (item for item in device.slots if item.start <= start < item.end), None
        )
        mode = covering.mode if covering is not None else None
        grid.append(_ScheduleSlot(start=start, end=stop, mode=mode, source="plan"))
        underlying.append(mode if mode is not None else last_mode)
    projected = _project_automation_override(grid, device.override, now=origin)
    spans = []
    for slot, release_mode in zip(projected, underlying, strict=True):
        assert slot is not None
        if comfort and slot.start < comfort_end and slot.source != "automation":
            assert latch_at is not None
            spans.append(
                ModeSpan(
                    start=slot.start,
                    end=slot.end,
                    mode=_SCHEDULE_OFF,
                    source="thermostat",
                    end_uncertain=True,
                    release_mode=release_mode,
                    elapsed_min=(slot.start - latch_at).total_seconds() / 60,
                )
            )
        elif slot.mode is not None:
            spans.append(
                ModeSpan(
                    start=slot.start, end=slot.end, mode=slot.mode, source=slot.source
                )
            )
    return tuple(spans)
