from unittest.mock import patch

import pandas as pd
import pytest

from kpower_forecast.core import KPowerForecast


@pytest.fixture
def sample_history():
    # 24 hours of data
    ds = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    # Simulate some production
    y = [0.0] * 6 + [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 8
    return pd.DataFrame({"ds": ds, "y": y})


def test_calibrate_efficiency(sample_history):
    kp = KPowerForecast(
        model_id="test_calib",
        latitude=46.0,
        longitude=14.5,
        forecast_type="solar",
        interval_minutes=60,
    )

    # Mock weather data within the dataframe
    df = sample_history.copy()
    df["cloud_cover"] = 10  # Clear sky
    df["shortwave_radiation"] = (
        [0.0] * 6 + [100, 200, 300, 400, 500, 500, 400, 300, 200, 100] + [0.0] * 8
    )

    factor = kp.calibrate_efficiency(df)

    # y = 1.0kWh per hour -> Power = 1.0kW
    # radiation = 100 W/m2
    # Factor = 1.0 / 100 = 0.01 (kW per W/m2)
    assert factor == pytest.approx(0.01)


@patch("kpower_forecast.core.WeatherClient")
@patch("kpower_forecast.core.ModelStorage")
def test_train_persistence(mock_storage_cls, mock_weather_cls, sample_history):
    mock_storage = mock_storage_cls.return_value
    mock_storage.load_model.return_value = None

    mock_weather = mock_weather_cls.return_value
    # Mock historical weather fetch
    weather_df = pd.DataFrame(
        {
            "ds": sample_history["ds"],
            "temperature_2m": 20.0,
            "cloud_cover": 10.0,
            "shortwave_radiation": 500.0,
        }
    )
    mock_weather.fetch_historical.return_value = weather_df
    mock_weather.resample_weather.side_effect = lambda df, _: df

    kp = KPowerForecast(model_id="test_persist", latitude=46.0, longitude=14.5)

    # Train
    kp.train(sample_history)

    # Verify storage.save_model was called with metadata
    assert mock_storage.save_model.called
    args, kwargs = mock_storage.save_model.call_args
    assert "metadata" in kwargs
    assert "efficiency_factor" in kwargs["metadata"]
