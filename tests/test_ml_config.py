import pytest

from kpower_forecast.ml import KPowerMLConfig, MLBackendType, MLForecastType
from kpower_forecast.ml.backends import registered_backends


def test_ml_config_defaults_to_nixtla_hybrid() -> None:
    config = KPowerMLConfig(model_id="ml", latitude=46.0, longitude=14.0)

    assert config.backend == MLBackendType.NIXTLA_HYBRID
    assert config.forecast_type == MLForecastType.SOLAR
    assert config.interval_levels == [50, 80, 90]


def test_ml_config_accepts_neuralforecast_backend() -> None:
    config = KPowerMLConfig(
        model_id="ml-neural",
        latitude=46.0,
        longitude=14.0,
        backend=MLBackendType.NEURALFORECAST,
    )

    assert config.backend == MLBackendType.NEURALFORECAST


def test_ml_config_rejects_invalid_interval_level() -> None:
    with pytest.raises(ValueError, match="between 1 and 99"):
        KPowerMLConfig(
            model_id="ml-invalid",
            latitude=46.0,
            longitude=14.0,
            interval_levels=[0, 90],
        )


def test_backend_registry_includes_nixtla_and_neuralforecast() -> None:
    assert registered_backends() == [
        MLBackendType.NEURALFORECAST,
        MLBackendType.NIXTLA_HYBRID,
    ]
