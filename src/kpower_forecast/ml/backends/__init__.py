"""Registered ML backend factories."""

from kpower_forecast.ml.config import MLBackendType

from .neural import NeuralForecastBackend
from .nixtla import NixtlaHybridBackend
from .registry import create_backend, register_backend, registered_backends

register_backend(MLBackendType.NIXTLA_HYBRID, NixtlaHybridBackend)
register_backend(MLBackendType.NEURALFORECAST, NeuralForecastBackend)

__all__ = [
    "NeuralForecastBackend",
    "NixtlaHybridBackend",
    "create_backend",
    "register_backend",
    "registered_backends",
]
