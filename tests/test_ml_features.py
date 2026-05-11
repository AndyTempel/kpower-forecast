import pandas as pd

from kpower_forecast.ml.config import KPowerMLConfig, MLForecastType
from kpower_forecast.ml.features import MLFeatureBuilder


def test_feature_builder_adds_calendar_physics_and_rolling_features() -> None:
    config = KPowerMLConfig(
        model_id="features",
        latitude=46.0,
        longitude=14.0,
        forecast_type=MLForecastType.HVAC,
        interval_minutes=60,
    )
    builder = MLFeatureBuilder(config)
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "temperature_2m": [10.0, 11.0, 12.0, 13.0],
            "shortwave_radiation": [0.0, 10.0, 20.0, 30.0],
            "wind_u_component_10m": [3.0, 3.0, 3.0, 3.0],
            "wind_v_component_10m": [4.0, 4.0, 4.0, 4.0],
        }
    )

    features = builder.build(df)

    expected = {
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "is_holiday",
        "clear_sky_ghi",
        "solar_elevation",
        "clear_sky_index",
        "wind_speed_10m",
        "temperature_2m_std_3h",
        "shortwave_radiation_std_6h",
        "hvac_temperature_delta",
    }
    assert expected.issubset(features.columns)
    assert features["wind_speed_10m"].tolist() == [5.0, 5.0, 5.0, 5.0]
