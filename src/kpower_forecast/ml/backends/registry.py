"""Backend registry for optional ML forecast engines."""

from collections.abc import Callable

from kpower_forecast.ml.config import KPowerMLConfig, MLBackendType

from .base import ForecastBackend

BackendFactory = Callable[[KPowerMLConfig], ForecastBackend]

_BACKENDS: dict[MLBackendType, BackendFactory] = {}


def register_backend(backend_type: MLBackendType, factory: BackendFactory) -> None:
    """Register a backend factory.

    Args:
        backend_type: Stable backend identifier.
        factory: Callable that builds a backend for a configuration.

    Returns:
        None.
    """
    _BACKENDS[backend_type] = factory


def create_backend(config: KPowerMLConfig) -> ForecastBackend:
    """Create the backend configured for ``config``.

    Args:
        config: ML runtime configuration.

    Returns:
        Backend instance.

    Raises:
        ValueError: If no factory is registered for the backend.
    """
    factory = _BACKENDS.get(config.backend)
    if factory is None:
        raise ValueError(f"Unsupported ML backend: {config.backend.value}")
    return factory(config)


def registered_backends() -> list[MLBackendType]:
    """Return registered backend identifiers."""
    return sorted(_BACKENDS, key=lambda backend: backend.value)
