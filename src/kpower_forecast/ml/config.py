"""Configuration models for the optional ML forecasting flow."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    @field_validator("interval_minutes")
    @classmethod
    def check_interval(cls, value: int) -> int:
        """Validate supported forecast grid intervals."""
        if value not in (15, 60):
            raise ValueError("interval_minutes must be 15 or 60")
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

    model_config = ConfigDict(arbitrary_types_allowed=True)
