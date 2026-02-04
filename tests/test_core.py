from unittest.mock import patch

import pandas as pd
import pytest

from kpower_forecast.core import KPowerForecast


@pytest.fixture
def mock_weather_client():
    with patch("kpower_forecast.core.WeatherClient") as mock:
        client_instance = mock.return_value

        # Mock fetch_historical
        client_instance.fetch_historical.return_value = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
                "temperature_2m": [20] * 24,
                "cloud_cover": [0] * 24,
                "shortwave_radiation": [500] * 24,
            }
        )

        # Mock resample
        def side_effect_resample(df, interval):
            # Simple pass through for mock
            return df

        client_instance.resample_weather.side_effect = side_effect_resample

        # Mock fetch_forecast
        client_instance.fetch_forecast.return_value = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC"),
                "temperature_2m": [20] * 24,
                "cloud_cover": [0] * 24,
                "shortwave_radiation": [500] * 24,
            }
        )

        yield client_instance


@pytest.fixture
def mock_storage():
    with patch("kpower_forecast.core.ModelStorage") as mock:
        yield mock.return_value


@pytest.fixture
def mock_prophet():
    with patch("kpower_forecast.core.Prophet") as mock:
        model_instance = mock.return_value
        # Mock predict return
        model_instance.predict.return_value = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC"),
                "yhat": [100] * 24,
            }
        )
        yield model_instance


def test_train(mock_weather_client, mock_storage, mock_prophet):
    kp = KPowerForecast("test_model", 0, 0)

    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            "y": [100] * 24,
        }
    )

    mock_storage.load_model.return_value = None  # No existing model

    kp.train(history)

    assert mock_weather_client.fetch_historical.called
    assert mock_prophet.fit.called
    assert mock_storage.save_model.called


def test_predict(mock_weather_client, mock_storage, mock_prophet):
    kp = KPowerForecast("test_model", 0, 0)

    mock_storage.load_model.return_value = mock_prophet

    forecast = kp.predict(days=1)

    assert mock_storage.load_model.called
    assert mock_weather_client.fetch_forecast.called
    assert not forecast.empty
    assert "yhat" in forecast.columns


def test_consumption_forecast(mock_weather_client, mock_storage, mock_prophet):
    kp = KPowerForecast(
        "test_consumption", 0, 0, forecast_type="consumption", heat_pump_mode=True
    )

    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            "y": [200] * 24,
        }
    )

    mock_storage.load_model.return_value = None
    kp.train(history)

    # Verify add_regressor was called for temperature
    mock_prophet.add_regressor.assert_called_with("temperature_2m")
    assert mock_prophet.fit.called
