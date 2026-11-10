"""Validated electrical-HVAC inputs and in-memory ensemble outputs.

Only numpy, pandas and pydantic are needed. Full minute paths deliberately have
no JSON serializer; display paths never stand in for an ensemble in planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from itertools import pairwise
from types import MappingProxyType
from typing import Literal, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

HVAC_CONTRACT_VERSION: int = 1
HVAC_ESTIMATOR_VERSION: str = "2"
FloatArray = NDArray[np.float64]
Diagnostic = Mapping[str, JsonValue]


class HvacError(ValueError):
    """Invalid HVAC input or unavailable model."""


class HvacTimeout(TimeoutError):
    """A fit or forecast exhausted its wall-clock budget."""


class HvacCancelled(RuntimeError):
    """A caller cancelled a fit or forecast before installation."""


class HvacFamily(str, Enum):
    """Compressor families; unsupported cycle families use energy-only fallback."""

    FIXED_SPEED = "fixed_speed"
    TWO_STAGE = "two_stage"
    MULTI_STAGE = "multi_stage"
    INVERTER = "inverter"
    UNKNOWN = "unknown"


class MeteredScope(str, Enum):
    """Electrical components actually measured by the input meter."""

    WHOLE_UNIT = "whole_unit"
    COMPRESSOR_ONLY = "compressor_only"
    UNKNOWN = "unknown"


class InputTier(str, Enum):
    """Evidence used by the forecast, independent of its reliability verdict."""

    T3_CONTROLLER = "t3_controller"
    T2_WATER = "t2_water"
    T1_POWER = "t1_power"
    T0_ENERGY = "t0_energy"


class HvacRegime(str, Enum):
    """Operating regimes, including reserved non-heating regimes."""

    SPACE_HEATING = "space_heating"
    DHW = "dhw"
    COOLING = "cooling"
    DEFROST = "defrost"
    STANDBY = "standby"
    UNKNOWN = "unknown"


class SojournEnd(str, Enum):
    """Observed reason for the right boundary of a spell."""

    NATURAL = "natural"
    COMMAND = "command"
    GAP = "gap"
    WINDOW = "window"
    OFF = "off"


ModeSource = Literal[
    "plan", "thermostat", "automation", "manual", "protection", "experiment", "other"
]


def utc(value: datetime) -> datetime:
    """Normalize an aware datetime; reject an ambiguous local timestamp.

    Args:
        value: An explicitly zoned instant.

    Returns:
        The same instant in UTC.

    Raises:
        ValueError: If the timestamp is naive.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise HvacError("HVAC timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


class _Input(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def _timestamps(cls, value: object) -> object:
        return utc(value) if isinstance(value, datetime) else value


class RatedPoint(_Input):
    """Declared heat output and electrical input at one rating condition."""

    source_temp_c: float
    supply_temp_c: float
    heat_output_kw: float = Field(gt=0)
    electrical_input_kw: float = Field(gt=0)


class HvacCapabilities(_Input):
    """Equipment facts; missing facts remain unknown rather than site defaults."""

    family: HvacFamily
    stages: int | None = Field(default=None, ge=1, strict=True)
    metered_scope: MeteredScope = MeteredScope.UNKNOWN
    rated_points: tuple[RatedPoint, ...] = ()
    min_run_minutes: float | None = Field(default=None, ge=0)
    min_off_minutes: float | None = Field(default=None, ge=0)
    backup_heater_kw: float | None = Field(default=None, ge=0)
    backup_heater_max_outdoor_c: float | None = None
    run_threshold_w: float | None = Field(default=None, gt=0)
    fingerprint: str = Field(min_length=1)


class HvacModelConfig(_Input):
    """Public tunables; the caller supplies its measured ingestion lag."""

    model_id: str = Field(min_length=1)
    timezone: str
    reference_mode: str = Field(default="NORMAL", min_length=1)
    off_modes: frozenset[str] = frozenset({"OFF"})
    training_days: int = Field(default=28, gt=0, strict=True)
    n_paths: int = Field(default=256, ge=16, strict=True)
    representative_paths: Literal[16] = 16
    seed: int = Field(default=0, ge=0, strict=True)
    minute_gap_tolerance_min: int = Field(default=5, ge=1, strict=True)
    dip_merge_min: int = Field(default=3, ge=0, strict=True)
    min_run_blip_min: int = Field(default=3, ge=1, strict=True)
    command_match_min: int = Field(default=2, ge=0, strict=True)
    recency_half_life_days: float = Field(default=3.0, gt=0)
    outdoor_hold_min: int = Field(default=60, ge=0, strict=True)
    outdoor_invalid_grace_min: int = Field(default=15, ge=0, strict=True)
    state_max_age_min: int = Field(default=10, gt=0, strict=True)
    ingest_lag_min: int = Field(default=1, ge=0, strict=True)
    dispersion_inflation: float = Field(default=1.0, gt=0)

    @field_validator("timezone")
    @classmethod
    def _zone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @model_validator(mode="after")
    def _modes(self) -> HvacModelConfig:
        if self.reference_mode in self.off_modes or any(not m for m in self.off_modes):
            raise HvacError("reference mode must be running-capable and modes nonempty")
        return self


class HistoryPolicy(_Input):
    """Owning EMS settings passed intact, with no independent policy defaults.

    Settings retain their owning names/version. Library policy evaluation uses
    these resolved values; a missing setting fails instead of relaxing eligibility.
    """

    version: int = Field(ge=1, strict=True)
    scope: Literal["direct_device", "system_attributed_heating"]
    settings: dict[str, JsonValue]
    measured_valid_qualities: frozenset[str] = Field(min_length=1)
    last_fingerprint_change_at: AwareDatetime | None = None


class ModeSpan(_Input):
    """One known future mode, possibly with an uncertain comfort release."""

    start: AwareDatetime
    end: AwareDatetime
    mode: str = Field(min_length=1)
    source: ModeSource
    end_uncertain: bool = False
    release_mode: str | None = Field(default=None, min_length=1)
    elapsed_min: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> ModeSpan:
        if self.end <= self.start:
            raise HvacError("mode span end must follow start")
        return self


class GridInterval(_Input):
    """One requested interval, identified by start and duration in minutes."""

    start: AwareDatetime
    minutes: Literal[1, 5, 15, 30, 60]

    @property
    def end(self) -> datetime:
        """Return the exclusive interval end."""
        return self.start + timedelta(minutes=self.minutes)


def validate_grid(grid: tuple[GridInterval, ...]) -> None:
    """Reject an empty, overlapping or discontinuous output grid.

    Args:
        grid: Consecutive requested intervals.

    Raises:
        ValueError: If intervals are not contiguous and increasing.
    """
    if not grid or any(a.end != b.start for a, b in pairwise(grid)):
        raise HvacError("forecast grid must be nonempty, ordered and contiguous")


def validate_minute_index(frame: pd.DataFrame) -> None:
    """Validate timestamp identity without filling or qualifying observations.

    Args:
        frame: Input observations, potentially sparse.

    Raises:
        ValueError: For invalid power, timestamp identity or non-minute labels.
    """
    index = frame.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or str(index.tz) != "UTC"
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.floor("min").equals(index)
        or "power_w" not in frame
    ):
        raise HvacError("minutes require power_w on a sorted unique UTC minute index")
    power = pd.to_numeric(frame.power_w, errors="raise")
    if power.isin([np.inf, -np.inf]).any() or (power < 0).any():
        raise HvacError("electrical power must be nonnegative finite or missing")


@dataclass(frozen=True)
class HvacTrainingData:
    """Snapshot of observations, operating history and the resolved owner policy."""

    capabilities: HvacCapabilities
    minutes: pd.DataFrame
    mode_events: pd.DataFrame
    weather: pd.DataFrame
    epoch_start: datetime
    lineage: Mapping[str, str]
    history_policy: HistoryPolicy
    control_events: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        validate_minute_index(self.minutes)
        object.__setattr__(self, "epoch_start", utc(self.epoch_start))
        for name in ("minutes", "mode_events", "weather", "control_events"):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        object.__setattr__(self, "lineage", MappingProxyType(dict(self.lineage)))


@dataclass(frozen=True)
class HvacForecastRequest:
    """An origin-time request; availability is enforced by the model slicer."""

    origin: datetime
    recent: pd.DataFrame
    schedule: tuple[ModeSpan, ...]
    last_mode: str | None
    weather_forecast: pd.DataFrame
    grid: tuple[GridInterval, ...]
    deadline_s: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", utc(self.origin))
        validate_minute_index(self.recent)
        validate_grid(self.grid)
        if not np.isfinite(self.deadline_s) or self.deadline_s <= 0:
            raise HvacError("prediction deadline must be positive and finite")
        if any(a.end > b.start for a, b in pairwise(self.schedule)):
            raise HvacError("mode spans must be ordered and non-overlapping")
        object.__setattr__(self, "recent", self.recent.copy(deep=True))
        object.__setattr__(
            self, "weather_forecast", self.weather_forecast.copy(deep=True)
        )


@dataclass(frozen=True)
class MinutePaths:
    """All ensemble power and state paths; never serialize this object.

    Unknown running/starts (energy-only forecasts) are None, never zero bitmasks.
    Arrays are copied and made read-only to prevent a display consumer mutating
    the planner's ensemble.
    """

    start: datetime
    power_w: NDArray[np.float32]
    running: NDArray[np.bool_] | None
    starts: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", utc(self.start))
        power = np.array(self.power_w, dtype=np.float32, copy=True)
        if (
            power.ndim != 2
            or min(power.shape) < 1
            or not np.isfinite(power).all()
            or (power < 0).any()
        ):
            raise HvacError(
                "minute paths require finite nonnegative [path, minute] power"
            )
        power.setflags(write=False)
        object.__setattr__(self, "power_w", power)
        for name in ("running", "starts"):
            raw = getattr(self, name)
            if raw is not None:
                if (
                    np.asarray(raw).dtype != np.dtype(bool)
                    or np.shape(raw) != power.shape
                ):
                    raise HvacError(
                        "path state masks must be bool with the power shape"
                    )
                value = np.array(raw, dtype=bool, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        if self.running is None and self.starts is not None:
            raise HvacError("starts cannot be known when running is unknown")

    @property
    def end(self) -> datetime:
        """Return the exclusive last minute boundary."""
        return self.start + timedelta(minutes=self.power_w.shape[1])


@dataclass(frozen=True)
class IntervalStats:
    """Statistics recomputed from all paths for one explicitly identified grid."""

    grid: tuple[GridInterval, ...]
    expected_kwh: FloatArray
    q05_kwh: FloatArray
    q10_kwh: FloatArray
    q50_kwh: FloatArray
    q90_kwh: FloatArray
    q95_kwh: FloatArray
    run_probability: FloatArray | None
    run_fraction: FloatArray | None
    start_probability: FloatArray | None
    peak_minute_w_q90: FloatArray


@dataclass(frozen=True)
class FitResult:
    """A candidate fit; returning it never installs or saves it."""

    fitted_at: datetime
    fingerprint: str
    tier: InputTier
    family: HvacFamily
    parameters: Mapping[str, JsonValue]
    diagnostics: Diagnostic
    fallback_reason: str | None = None
    estimator_version: str = HVAC_ESTIMATOR_VERSION


@dataclass(frozen=True)
class HvacForecast:
    """Full ensemble and separately marked representative display paths."""

    origin: datetime
    grid: tuple[GridInterval, ...]
    intervals: IntervalStats
    minute_paths: MinutePaths
    paths_w: NDArray[np.float32]
    display_index: int | None
    tier: InputTier
    family: HvacFamily
    fallback_reason: str | None
    diagnostics: Diagnostic
    fingerprint: str
    estimator_version: str = HVAC_ESTIMATOR_VERSION

    def stats(self, grid: tuple[GridInterval, ...]) -> IntervalStats:
        """Recompute grid statistics from all retained paths.

        Args:
            grid: Contiguous intervals within the retained minute horizon.

        Returns:
            Full-ensemble statistics on that exact grid.

        Raises:
            ValueError: If the grid exceeds the path horizon.
        """
        from .aggregate import interval_stats

        return interval_stats(self.minute_paths, grid)
