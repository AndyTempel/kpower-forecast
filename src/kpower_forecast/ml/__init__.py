"""Optional ML forecasting add-on for KPower Forecast.

The root :mod:`kpower_forecast` package intentionally does not import this
module so base installations remain lightweight.
"""

from .alignment import FORECAST_CONTRACT_VERSION, ForecastAlignmentError
from .baselines import BASELINE_NAME, BaselineMetrics, evaluate_local_slot_baseline
from .config import KPowerMLConfig, MLBackendType, MLForecastType
from .forecast import KPowerMLForecast, bridge_short_target_gaps

__all__ = [
    "ForecastAlignmentError",
    "FORECAST_CONTRACT_VERSION",
    "BASELINE_NAME",
    "BaselineMetrics",
    "KPowerMLConfig",
    "KPowerMLForecast",
    "MLBackendType",
    "MLForecastType",
    "bridge_short_target_gaps",
    "evaluate_local_slot_baseline",
]
