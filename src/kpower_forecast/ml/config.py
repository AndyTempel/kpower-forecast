"""Configuration models for the optional ML forecasting flow."""

from enum import Enum
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kpower_forecast.core import DataCategory, MeasurementUnit


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
    interval_levels: list[int] = Field(default_factory=lambda: [50, 80, 90])
    holiday_country: Optional[str] = None
    holiday_subdivision: Optional[str] = None
    calibration_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    adaptive_weather_correction: bool = True
    min_weather_correction_samples: int = Field(default=8, gt=0)
    inverter_ac_limit_kw: Optional[float] = Field(default=None, gt=0)
    grid_export_limit_kw: Optional[float] = Field(default=None, gt=0)
    # Holdout selection between the ML model, a degree-hour regression and the
    # slot/weekday median (see kpower_forecast.ml.selection). Off by default.
    candidate_selection: bool = False
    min_selection_holdout_rows: int = Field(default=96, gt=0)
    regression_base_temperature_c: float = 16.0
    regression_extra_features: list[str] = Field(
        default_factory=lambda: ["shortwave_radiation"]
    )

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
        """Reject candidate selection for solar targets.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If selection is enabled for a solar forecast, whose
                curtailment and night constraints assume the ML output.
        """
        if self.candidate_selection and self.forecast_type == MLForecastType.SOLAR:
            raise ValueError("candidate_selection is not supported for solar")
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)
