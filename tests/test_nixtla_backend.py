import pandas as pd

from kpower_forecast.ml.backends.nixtla import NixtlaHybridBackend
from kpower_forecast.ml.config import KPowerMLConfig, MLForecastType


def test_non_solar_targets_do_not_fit_solar_radiation_baseline() -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=60,
            forecast_type=MLForecastType.CONSUMPTION,
        )
    )
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2026-05-01", periods=24, freq="h", tz="UTC"),
            "y": [0.4] * 24,
        }
    )
    features = pd.DataFrame(
        {
            "ds": history["ds"],
            "shortwave_radiation": [0.0] * 6 + [500.0] * 12 + [0.0] * 6,
        }
    )

    backend._fit_solar_profile(history, features)

    assert backend._solar_global_factor is None
    assert backend._solar_profile == {}
    assert backend._predict_solar_baseline(features) is None


def test_solar_target_fits_solar_radiation_baseline() -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="solar",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=60,
            forecast_type=MLForecastType.SOLAR,
        )
    )
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2026-05-01", periods=24, freq="h", tz="UTC"),
            "y": [0.0] * 6 + [0.5] * 12 + [0.0] * 6,
        }
    )
    features = pd.DataFrame(
        {
            "ds": history["ds"],
            "shortwave_radiation": [0.0] * 6 + [500.0] * 12 + [0.0] * 6,
        }
    )

    backend._fit_solar_profile(history, features)

    assert backend._solar_global_factor is not None
    assert backend._solar_profile
    assert backend._predict_solar_baseline(features) is not None
