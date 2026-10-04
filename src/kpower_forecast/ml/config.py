"""Configuration models for the optional ML forecasting flow."""

from enum import Enum
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kpower_forecast.core import DataCategory, MeasurementUnit
from kpower_forecast.ml.selection import ML_CANDIDATE, SELECTION_CANDIDATES


class MLForecastType(str, Enum):
    """Forecast targets supported by the ML add-on."""

    SOLAR = "solar"
    CONSUMPTION = "consumption"
    HVAC = "hvac"


class MLBackendType(str, Enum):
    """Backend identifiers for optional ML forecasters."""

    NIXTLA_HYBRID = "nixtla_hybrid"
    NEURALFORECAST = "neuralforecast"
    FOUNDATION = "foundation"


class HybridStructure(str, Enum):
    """How the Nixtla hybrid backend forms its structural forecast.

    ``recursive_seasonal_naive`` repeats the last observed day and corrects it
    with a LightGBM residual model fed its own lags recursively. That suits
    smooth series but replays one-off events (e.g. heat-pump runs) into every
    future day and can oscillate over multi-day horizons.

    ``profile_direct`` uses a multi-day mean profile per local slot and
    weekday class and predicts the residual directly from weather and
    calendar features for every future row, with no lags or recursion.
    """

    RECURSIVE_SEASONAL_NAIVE = "recursive_seasonal_naive"
    PROFILE_DIRECT = "profile_direct"


class KPowerMLConfig(BaseModel):
    """Runtime configuration for the optional ML forecasting model."""

    model_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    storage_path: str = "./data"
    interval_minutes: int = Field(default=15)
    timezone: str = "UTC"
    preserve_gaps: bool = False
    max_bridged_gap_intervals: int = Field(default=0, ge=0, le=16)
    forecast_type: MLForecastType = MLForecastType.SOLAR
    data_category: DataCategory = DataCategory.INSTANT_ENERGY
    unit: MeasurementUnit = MeasurementUnit.KWH
    backend: MLBackendType = MLBackendType.NIXTLA_HYBRID
    backend_params: dict[str, Any] = Field(default_factory=dict)
    hybrid_structure: HybridStructure = HybridStructure.RECURSIVE_SEASONAL_NAIVE
    profile_lookback_days: int = Field(default=28, ge=1, le=366)
    # profile_direct: each weekday-class profile is shrunk toward the all-days
    # profile with weight n / (n + prior), n = days of that class in the
    # lookback, so a class seen on few days cannot replay their events.
    profile_class_prior_days: float = Field(default=4.0, ge=0.0, le=366.0)
    # profile_direct: half-width of a centred circular moving average over the
    # profile's local time of day (± minutes; 0 = off), a multiple of the
    # interval. Compressor runs do not repeat to the minute.
    profile_smoothing_minutes: int = Field(default=0, ge=0, le=180)
    interval_levels: list[int] = Field(default_factory=lambda: [50, 80, 90])
    holiday_country: Optional[str] = None
    holiday_subdivision: Optional[str] = None
    calibration_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    adaptive_weather_correction: bool = True
    min_weather_correction_samples: int = Field(default=8, gt=0)
    inverter_ac_limit_kw: Optional[float] = Field(default=None, gt=0)
    grid_export_limit_kw: Optional[float] = Field(default=None, gt=0)
    # Rolling-origin selection between the ML model and transparent
    # benchmarks (see kpower_forecast.ml.selection). Off by default.
    candidate_selection: bool = False
    # Candidates that may be served; the ML model is always scored.
    selection_candidates: list[str] = Field(
        default_factory=lambda: list(SELECTION_CANDIDATES)
    )
    # Day-ahead origins at local midnight over the most recent days.
    selection_backtest_days: int = Field(default=7, ge=1, le=60)
    selection_horizon_hours: int = Field(default=24, ge=1, le=168)
    selection_min_origins: int = Field(default=3, ge=1, le=60)
    # A candidate whose mean error exceeds this share of mean actual load (or
    # the floor, whichever is larger) is not eligible to win.
    selection_bias_tolerance: float = Field(default=0.15, gt=0.0, le=1.0)
    selection_bias_floor_kw: float = Field(default=0.05, ge=0.0)
    regression_base_temperature_c: float = 16.0
    regression_extra_features: list[str] = Field(
        default_factory=lambda: ["shortwave_radiation"]
    )
    # Caller-supplied numeric columns known in advance (e.g. a scheduled HVAC
    # mode). Required in training history and, for every model-grid row, in
    # ``known_future`` at prediction.
    known_covariates: list[str] = Field(default_factory=list)

    @field_validator("interval_minutes")
    @classmethod
    def check_interval(cls, value: int) -> int:
        """Validate supported forecast grid intervals."""
        if value not in (15, 60):
            raise ValueError("interval_minutes must be 15 or 60")
        return value

    @field_validator("timezone")
    @classmethod
    def check_timezone(cls, value: str) -> str:
        """Require an explicit valid IANA timezone name."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"invalid IANA timezone: {value}") from exc
        return value

    @field_validator("interval_levels")
    @classmethod
    def check_interval_levels(cls, value: list[int]) -> list[int]:
        """Validate conformal interval coverage levels."""
        if not value:
            raise ValueError("interval_levels must not be empty")
        unique_levels = sorted(set(value))
        for level in unique_levels:
            if level <= 0 or level >= 100:
                raise ValueError("interval levels must be between 1 and 99")
        return unique_levels

    @model_validator(mode="after")
    def check_gap_bridging(self) -> "KPowerMLConfig":
        """Reject gap bridging where it would be ineffective or bias the model.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If bridging is enabled without ``preserve_gaps``, for
                cumulative-energy input, or for solar targets.
        """
        if self.max_bridged_gap_intervals == 0:
            return self
        if not self.preserve_gaps:
            raise ValueError("max_bridged_gap_intervals requires preserve_gaps=True")
        if self.data_category == DataCategory.CUMULATIVE_ENERGY:
            # One dropped meter reading removes two interval deltas, and a
            # linear bridge would not conserve the known meter delta.
            raise ValueError(
                "max_bridged_gap_intervals does not support cumulative_energy input"
            )
        if self.forecast_type == MLForecastType.SOLAR:
            # The solar radiation profile is calibrated from training targets.
            raise ValueError("max_bridged_gap_intervals is not supported for solar")
        return self

    @model_validator(mode="after")
    def check_candidate_selection(self) -> "KPowerMLConfig":
        """Validate candidate selection and degree-hour regression inputs.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If selection is enabled for a solar forecast, whose
                curtailment and night constraints assume the ML output, or if
                regression extras repeat or name reserved columns.
        """
        if self.candidate_selection and self.forecast_type == MLForecastType.SOLAR:
            raise ValueError("candidate_selection is not supported for solar")
        unknown = set(self.selection_candidates) - set(SELECTION_CANDIDATES)
        if unknown:
            raise ValueError(f"unknown selection_candidates: {sorted(unknown)}")
        if ML_CANDIDATE not in self.selection_candidates:
            # The ML model is the reference every benchmark must beat.
            raise ValueError(f"selection_candidates must include {ML_CANDIDATE!r}")
        if len(set(self.selection_candidates)) != len(self.selection_candidates):
            raise ValueError("selection_candidates must be unique")
        if self.selection_min_origins > self.selection_backtest_days:
            raise ValueError(
                "selection_min_origins must not exceed selection_backtest_days"
            )
        if self.profile_smoothing_minutes % self.interval_minutes:
            raise ValueError(
                "profile_smoothing_minutes must be a multiple of interval_minutes"
            )
        if self.hybrid_structure == HybridStructure.PROFILE_DIRECT:
            if self.forecast_type == MLForecastType.SOLAR:
                # Solar uses its radiation profile baseline.
                raise ValueError("profile_direct hybrid structure is not for solar")
            if self.backend != MLBackendType.NIXTLA_HYBRID:
                # Other backends would silently ignore the requested structure.
                raise ValueError(
                    "profile_direct hybrid structure requires the nixtla_hybrid backend"
                )
        reserved = {"ds", "y", "temperature_2m"}.intersection(
            self.regression_extra_features
        )
        if reserved:
            # ``y`` would leak the holdout target into selection.
            raise ValueError(
                f"regression_extra_features must not include {sorted(reserved)}"
            )
        if len(set(self.regression_extra_features)) != len(
            self.regression_extra_features
        ):
            raise ValueError("regression_extra_features must be unique")
        # unique_id is the Nixtla series key.
        reserved_covariates = {"ds", "y", "unique_id"}.intersection(
            self.known_covariates
        )
        if reserved_covariates:
            raise ValueError(
                f"known_covariates must not include {sorted(reserved_covariates)}"
            )
        if len(set(self.known_covariates)) != len(self.known_covariates):
            raise ValueError("known_covariates must be unique")
        if self.known_covariates and self.backend != MLBackendType.NIXTLA_HYBRID:
            # Other backends do not read exogenous features; the inputs would
            # be required yet silently ignored.
            raise ValueError("known_covariates require the nixtla_hybrid backend")
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)
