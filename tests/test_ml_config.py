import pytest

from kpower_forecast.core import DataCategory
from kpower_forecast.ml import KPowerMLConfig, MLBackendType, MLForecastType
from kpower_forecast.ml.backends import registered_backends


def test_ml_config_defaults_to_nixtla_hybrid() -> None:
    config = KPowerMLConfig(model_id="ml", latitude=46.0, longitude=14.0)

    assert config.backend == MLBackendType.NIXTLA_HYBRID
    assert config.forecast_type == MLForecastType.SOLAR
    assert config.interval_levels == [50, 80, 90]
    assert config.timezone == "UTC"


def test_ml_config_validates_timezone() -> None:
    config = KPowerMLConfig(
        model_id="local",
        latitude=46.0,
        longitude=14.0,
        timezone="Europe/Ljubljana",
    )
    assert config.timezone == "Europe/Ljubljana"

    with pytest.raises(ValueError, match="invalid IANA timezone"):
        KPowerMLConfig(
            model_id="bad", latitude=46.0, longitude=14.0, timezone="Mars/Base"
        )


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"preserve_gaps": False}, "requires preserve_gaps"),
        ({"data_category": DataCategory.CUMULATIVE_ENERGY}, "cumulative_energy"),
        ({"forecast_type": MLForecastType.SOLAR}, "not supported for solar"),
    ],
)
def test_ml_config_rejects_gap_bridging_where_it_would_mislead(
    overrides: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "model_id": "hvac",
        "latitude": 46.0,
        "longitude": 14.0,
        "preserve_gaps": True,
        "forecast_type": MLForecastType.HVAC,
        "data_category": DataCategory.POWER,
        "max_bridged_gap_intervals": 4,
    }
    assert KPowerMLConfig(**base).max_bridged_gap_intervals == 4  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        KPowerMLConfig(**{**base, **overrides})  # type: ignore[arg-type]
