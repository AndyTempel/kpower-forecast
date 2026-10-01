"""Passive first-order building-temperature identification and prediction."""

from .model import (
    KPowerThermalForecast,
    ThermalModelConfig,
    ThermalModelDiagnostics,
    ThermalObservedTransition,
    ThermalPrediction,
    ThermalPredictionInterval,
    ThermalTrainingTransition,
)
from .naive import NAIVE_SOURCE, predict_naive, recent_trend_c_per_hour

__all__ = [
    "NAIVE_SOURCE",
    "KPowerThermalForecast",
    "ThermalModelConfig",
    "ThermalModelDiagnostics",
    "ThermalObservedTransition",
    "ThermalPrediction",
    "ThermalPredictionInterval",
    "ThermalTrainingTransition",
    "predict_naive",
    "recent_trend_c_per_hour",
]
