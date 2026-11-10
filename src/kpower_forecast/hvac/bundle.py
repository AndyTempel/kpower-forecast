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
from typing import Literal, Mapping, Self, cast, overload
from zoneinfo import ZoneInfo

import numpy as np
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

from .types import (
    HistoryPolicy,
    HvacCapabilities,
    HvacModelConfig,
    HvacTrainingData,
    ModeSource,
    ModeSpan,
    utc,
)


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

    @overload
    def available_at(
        self, origin: datetime, *, include_inputs: Literal[False] = False
    ) -> pd.DataFrame: ...

    @overload
    def available_at(
        self, origin: datetime, *, include_inputs: Literal[True]
    ) -> BundleSnapshot: ...

    def available_at(
        self, origin: datetime, *, include_inputs: bool = False
    ) -> pd.DataFrame | BundleSnapshot:
        """Return an isolated observed-history slice after the availability boundary.

        Args:
            origin: Aware UTC origin of a fit or prediction.
            include_inputs: Return the complete separated feature/outcome snapshot
                for scoring. The default preserves the original minute-only API.

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
        minutes = frame.loc[frame.index <= cutoff].copy(deep=True)
        return _bundle_snapshot(self, origin, minutes) if include_inputs else minutes


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


@dataclass(frozen=True)
class BundleSnapshot:
    """One availability boundary: causal features separated from scoring outcomes.

    Outcomes are never passed into fit or predict. S1 deliberately reads oracle
    commands from outcomes and must be reported as such. Day-0 weather is an
    explicitly optimistic development approximation, never archived evidence.
    """

    origin: datetime
    training: HvacTrainingData
    weather_forecast: pd.DataFrame
    vintage_kinds: tuple[str, ...]
    schedule_payload: Mapping[str, JsonValue] | None
    schedule_hash: str | None
    last_mode: str | None
    last_mode_event: datetime | None
    ledger: pd.DataFrame
    baselines: Mapping[str, pd.DataFrame]
    outcomes: Mapping[str, pd.DataFrame]
    provenance: Mapping[str, JsonValue]

    def spans(
        self,
        variant: Literal["s0", "s1", "s2", "s3"],
        *,
        device_id: str,
        horizon_end: datetime,
    ) -> tuple[ModeSpan, ...]:
        """Build held, explicitly oracle, or archived effective future spans.

        Args:
            variant: S0 holds current mode; S1 uses actual future commands; S2
                uses only the archived version available at this origin; S3 is
                the development proxy for the EMS plan: future ``plan`` commands
                only, and an active comfort/automation OFF released at an
                uncertain time into the latest planned mode (EMS policy E4.2).
            device_id: Selected heat pump for S2.
            horizon_end: Requested output horizon.

        Returns:
            Known future spans, with uncertainty only for an active S2 latch.

        Raises:
            ValueError: For an unknown variant or malformed schedule evidence.
        """
        if variant == "s0":
            return ()
        if variant == "s2":
            return effective_mode_spans(
                self.schedule_payload,
                device_id,
                self.origin,
                self.last_mode,
                self.last_mode_event,
                horizon_end=horizon_end,
            )
        if variant not in ("s1", "s3"):
            raise ValueError("unknown schedule variant")
        events = self.outcomes.get("mode_events.parquet", pd.DataFrame())
        if events.empty:
            return ()
        if variant == "s3":
            return self._plan_proxy_spans(events, horizon_end)
        future = events.loc[
            (events.time > self.origin) & (events.time < horizon_end)
        ].sort_values("time", kind="stable")
        spans = []
        for i, row in enumerate(future.itertuples()):
            end = (
                cast(pd.Timestamp, future.time.iloc[i + 1]).to_pydatetime()
                if i + 1 < len(future)
                else horizon_end
            )
            spans.append(
                ModeSpan(
                    start=cast(pd.Timestamp, row.time).to_pydatetime(),
                    end=end,
                    mode=str(row.mode),
                    source=cast(ModeSource, str(row.source)),
                )
            )
        return tuple(spans)

    def _plan_proxy_spans(
        self, events: pd.DataFrame, horizon_end: datetime
    ) -> tuple[ModeSpan, ...]:
        past = events.loc[events.time <= self.origin].sort_values("time", kind="stable")
        future = events.loc[
            (events.time > self.origin)
            & (events.time < horizon_end)
            & events.source.eq("plan")
        ].sort_values("time", kind="stable")
        plan_past = past.loc[past.source.eq("plan") & ~past["mode"].eq("OFF")]
        planned = str(plan_past["mode"].iloc[-1]) if len(plan_past) else None
        spans: list[ModeSpan] = []
        boundaries = [cast(pd.Timestamp, t).to_pydatetime() for t in future.time]
        first_end = boundaries[0] if boundaries else horizon_end
        if len(past):
            last = past.iloc[-1]
            if (
                str(last["mode"]) == "OFF"
                and str(last.source) in ("thermostat", "automation")
                and planned is not None
            ):
                started = cast(pd.Timestamp, last.time).to_pydatetime()
                spans.append(
                    ModeSpan(
                        start=self.origin,
                        end=first_end,
                        mode="OFF",
                        source=cast(ModeSource, str(last.source)),
                        end_uncertain=True,
                        release_mode=planned,
                        # Age of the latch at span start (the origin).
                        elapsed_min=(self.origin - started).total_seconds() / 60,
                    )
                )
        for i, row in enumerate(future.itertuples()):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else horizon_end
            spans.append(
                ModeSpan(
                    start=boundaries[i],
                    end=end,
                    mode=str(row.mode),
                    source="plan",
                )
            )
        return tuple(spans)


def _utc_column(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    if name not in frame:
        raise ValueError(f"missing availability column: {name}")
    index = pd.DatetimeIndex(frame[name])
    if str(index.tz) != "UTC" or index.hasnans:
        raise ValueError(f"{name} must contain aware UTC timestamps")
    return index


def _causal_events(frame: pd.DataFrame, origin: datetime, time: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    known = _utc_column(frame, time) <= pd.Timestamp(origin)
    if "available_at" in frame:
        known &= _utc_column(frame, "available_at") <= pd.Timestamp(origin)
    return cast(pd.DataFrame, frame.loc[known].copy().sort_values(time, kind="stable"))


def _origin_weather(frame: pd.DataFrame, origin: datetime) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    source = frame.copy()
    valid = _utc_column(source, "valid_at")
    available = _utc_column(source, "available_at")
    allowed = {"archived", "approx_previous_runs", "approx_day0"}
    if not source.vintage_kind.isin(allowed).all():
        raise ValueError("unknown weather vintage kind")
    optimistic = source.vintage_kind.eq("approx_day0").to_numpy()
    known = (available <= pd.Timestamp(origin)) | optimistic
    previous = source.vintage_kind.eq("approx_previous_runs").to_numpy()
    required_day = np.maximum(
        np.ceil((valid - pd.Timestamp(origin)).total_seconds() / 86400), 1
    )
    implied_day = (valid - available).total_seconds() / 86400
    known &= ~previous | np.isclose(implied_day, required_day)
    source = source.loc[known].copy()
    source["_priority"] = source.vintage_kind.map(
        {"archived": 0, "approx_previous_runs": 1, "approx_day0": 2}
    )
    source = source.sort_values(
        ["valid_at", "_priority", "available_at"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates("valid_at")
    # Day0 deliberately has no causal vintage. Preserve its original stamp and
    # mark a virtual origin-time availability, never mislabel it as an archive.
    day0 = source.vintage_kind.eq("approx_day0")
    source["original_available_at"] = source.available_at
    source.loc[day0, "available_at"] = pd.Timestamp(origin)
    return source.drop(columns="_priority").set_index("valid_at").sort_index()


def _origin_schedule(
    frame: pd.DataFrame, origin: datetime
) -> tuple[Mapping[str, JsonValue] | None, str | None]:
    if frame.empty:
        return None, None
    at = _utc_column(frame, "available_at")
    until = _utc_column(frame, "valid_until")
    eligible = frame.loc[(at <= pd.Timestamp(origin)) & (pd.Timestamp(origin) < until)]
    if eligible.empty:
        return None, None
    latest = eligible.sort_values("available_at", kind="stable").iloc[-1]
    payload = json.loads(str(latest.payload_json), parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("schedule version must be a JSON object")
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at", "valid_until"}
    }
    digest = hashlib.sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        "content_hash" in latest
        and pd.notna(latest.content_hash)
        and str(latest.content_hash) != digest
    ):
        raise ValueError("schedule semantic content hash mismatch")
    # A version retains its first whole payload; a seen heartbeat supplies current
    # publication/expiry times without changing that version's content identity.
    if "generated_at" in latest and pd.notna(latest.generated_at):
        generated = pd.Timestamp(latest.generated_at)
        if str(generated.tz) != "UTC":
            raise ValueError("schedule generated_at must be aware UTC")
        payload["generated_at"] = generated.isoformat()
    payload["valid_until"] = pd.Timestamp(latest.valid_until).isoformat()
    _SchedulePayload.model_validate(payload)
    return MappingProxyType(payload), digest


def _bundle_snapshot(
    bundle: HvacEvaluationBundle, origin: datetime, minutes: pd.DataFrame
) -> BundleSnapshot:
    from .frames import prepare_minutes

    manifest = bundle.manifest
    policy_document = bundle.documents.get("history_policy.json")
    qualification = bundle.documents.get("minute_qualification.json")
    if not isinstance(policy_document, dict) or not isinstance(qualification, dict):
        raise ValueError(
            "full bundle inputs require resolved policy and minute quality evidence"
        )
    scope = qualification.get("scope")
    if scope not in {
        "direct_device_minutes",
        "direct_device",
        "system_attributed_heating",
    }:
        raise ValueError("unknown minute target scope")
    settings = policy_document.get(
        "minute_history_policy", policy_document.get("policy")
    )
    if not isinstance(settings, dict):
        raise ValueError("missing resolved minute history policy")
    state = qualification.get("model_state")
    last_change = (
        state.get("last_fingerprint_change_at") if isinstance(state, dict) else None
    )
    policy = HistoryPolicy.model_validate(
        {
            "version": settings.get("_HISTORY_POLICY_VERSION"),
            "scope": (
                "direct_device"
                if scope in {"direct_device_minutes", "direct_device"}
                else scope
            ),
            "settings": settings,
            "measured_valid_qualities": policy_document.get("measured_valid_qualities"),
            "last_fingerprint_change_at": last_change,
        }
    )
    events = _causal_events(
        bundle.tables.get("mode_events.parquet", pd.DataFrame()), origin, "time"
    )
    controls = _causal_events(
        bundle.tables.get("control_events.parquet", pd.DataFrame()), origin, "time"
    )
    weather = bundle.tables.get("weather.parquet", pd.DataFrame()).copy()
    if len(weather):
        stamp = (
            _utc_column(weather, "ds")
            if "ds" in weather
            else _utc_index(weather.index, "weather")
        )
        known = stamp <= pd.Timestamp(origin)
        if "available_at" in weather:
            known &= _utc_column(weather, "available_at") <= pd.Timestamp(origin)
        weather = cast(pd.DataFrame, weather.loc[known].copy())
    lineage = {"controller": manifest.controller_id, "device": manifest.device_id}
    for column in minutes:
        if isinstance(column, str) and column.endswith("__source_device"):
            signal = column.removesuffix("__source_device")
            valid = minutes.get(
                signal, pd.Series(index=minutes.index, dtype=float)
            ).notna()
            valid &= minutes.get(
                signal + "__measured_valid", pd.Series(False, index=minutes.index)
            ).eq(True)
            if signal + "__quality" in minutes:
                valid &= minutes[signal + "__quality"].isin(
                    policy.measured_valid_qualities
                )
            known_source = minutes.loc[valid, column].dropna()
            if len(known_source):
                lineage[column] = str(known_source.iloc[-1])
    training = HvacTrainingData(
        HvacCapabilities.model_validate(bundle.documents["capabilities.json"]),
        minutes,
        events,
        weather,
        manifest.start,
        lineage,
        policy,
        controls,
    )
    forecast_weather = _origin_weather(
        bundle.tables.get("weather_vintages.parquet", pd.DataFrame()), origin
    )
    kinds = (
        tuple(sorted(str(v) for v in forecast_weather.vintage_kind.unique()))
        if len(forecast_weather)
        else ()
    )
    payload, digest = _origin_schedule(
        bundle.tables.get("schedules.parquet", pd.DataFrame()), origin
    )
    last_mode = str(events["mode"].iloc[-1]) if len(events) else None
    latch = (
        events.loc[events["mode"].eq("OFF") & events.source.eq("thermostat")]
        if len(events)
        else events
    )
    latch_at = latch.time.iloc[-1].to_pydatetime() if len(latch) else None
    ledger = _causal_events(
        bundle.tables.get("ledger.parquet", pd.DataFrame()), origin, "observed_until"
    )
    baselines = {
        name: _causal_events(bundle.tables.get(name, pd.DataFrame()), origin, "origin")
        for name in (
            "served_forecast.parquet",
            "r1c_replay.parquet",
            "legacy_shadow.parquet",
            "r1c_shadow.parquet",
            "legacy_awareness.parquet",
        )
    }
    # Actuals are separate from features and are qualified at export availability.
    # They can contain future outcomes relative to this prediction origin.
    outcome_cfg = HvacModelConfig(
        model_id="actuals",
        timezone=manifest.timezone,
        ingest_lag_min=cast(int, manifest.ingest_lag_min),
    )
    actuals = {
        name: bundle.tables.get(name, pd.DataFrame()).copy(deep=True)
        for name in (
            "site_minutes.parquet",
            "site_quarters.parquet",
            "tariff_quarters.parquet",
            "mode_events.parquet",
            "live_forecasts.parquet",
            "live_first_hour.parquet",
            "live_horizon_energy.parquet",
            "live_reserves.parquet",
            "live_window_inputs.parquet",
        )
    }
    actuals["minutes.parquet"] = prepare_minutes(
        bundle.tables["minutes.parquet"],
        outcome_cfg,
        as_of=manifest.exported_at,
        policy=policy,
        training=False,
        epoch_start=manifest.start,
    )
    return BundleSnapshot(
        origin,
        training,
        forecast_weather,
        kinds,
        payload,
        digest,
        last_mode,
        latch_at,
        ledger,
        MappingProxyType(baselines),
        MappingProxyType(actuals),
        MappingProxyType(
            {
                "scope": manifest.metadata.get("scope", "development"),
                "epoch_basis": "bundle_start",
                "lineage_basis": "measured_valid_source_device_ids",
                "tariff_source": manifest.metadata.get("tariff_source"),
            }
        ),
    )
