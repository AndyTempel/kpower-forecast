"""Optional ML forecasting add-on for KPower Forecast.

The root :mod:`kpower_forecast` package intentionally does not import this
module so base installations remain lightweight.
"""

from .config import KPowerMLConfig, MLBackendType, MLForecastType
from .forecast import KPowerMLForecast

__all__ = [
    "KPowerMLConfig",
    "KPowerMLForecast",
    "MLBackendType",
    "MLForecastType",
]
