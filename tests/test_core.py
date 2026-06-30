import warnings
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest
from prophet import Prophet

from kpower_forecast.core import KPowerForecast
from kpower_forecast.utils import calculate_solar_elevation


@pytest.fixture
def sample_history():
    # 24 hours of data
    ds = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    # Simulate some production
    y = [0.0] * 6 + [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 8
    return pd.DataFrame({"ds": ds, "y": y})


def test_calculate_solar_elevation_uses_2026_without_pysolar_warning():
    times = pd.date_range("2026-06-30T10:00:00Z", periods=4, freq="h")

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        elevations = calculate_solar_elevation(46.0569, 14.5058, times)

    assert len(elevations) == 4
    assert elevations[0] > 0.0
    assert not any("Leap seconds" in str(item.message) for item in warning_records)


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


def test_calibrate_efficiency_does_not_apply_default_home_system_cap():
    kp = KPowerForecast(
        model_id="test_calib_no_default_cap",
        latitude=46.0,
        longitude=14.5,
        forecast_type="solar",
        interval_minutes=60,
    )

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-01 08:00", periods=12, freq="h", tz="UTC"),
            "y": [150.0] * 12,
            "cloud_cover": [0.0] * 12,
            "shortwave_radiation": [100.0] * 12,
        }
    )

    assert kp.calibrate_efficiency(df) == pytest.approx(1.5)


def test_calibrate_efficiency_profile_learns_morning_envelope():
    kp = KPowerForecast(
        model_id="test_profile",
        latitude=46.0,
        longitude=14.5,
        forecast_type="solar",
        interval_minutes=15,
    )

    timestamps = []
    production = []
    interval_hours = 0.25
    for day in pd.date_range("2024-06-01", periods=20, freq="D", tz="UTC"):
        for minute_of_day in range(5 * 60, 21 * 60, 15):
            timestamp = day + pd.Timedelta(minutes=minute_of_day)
            efficiency = 0.06 if minute_of_day == 6 * 60 else 0.02
            timestamps.append(timestamp)
            production.append(efficiency * 100.0 * interval_hours)

    df = pd.DataFrame(
        {
            "ds": timestamps,
            "y": production,
            "cloud_cover": [0.0] * len(timestamps),
            "shortwave_radiation": [100.0] * len(timestamps),
            "snow_depth": [0.0] * len(timestamps),
        }
    )

    kp.config.efficiency_factor = kp.calibrate_efficiency(df)
    profile = kp.calibrate_efficiency_profile(df)

    assert profile is not None
    assert kp.config.efficiency_factor == pytest.approx(0.02)
    assert profile[6 * 60] == pytest.approx(0.06)


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
    assert "efficiency_profile" in kwargs["metadata"]


class DummyForecastModel:
    """Minimal Prophet-like model for post-processing tests."""

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Return deliberately high values so physical caps are observable."""
        return pd.DataFrame(
            {
                "ds": future["ds"],
                "yhat": [10.0] * len(future),
                "yhat_lower": [10.0] * len(future),
                "yhat_upper": [10.0] * len(future),
            }
        )


def test_predict_uses_efficiency_profile_for_cap(monkeypatch, tmp_path):
    kp = KPowerForecast(
        model_id="test_profile_cap",
        latitude=46.0,
        longitude=14.5,
        storage_path=str(tmp_path),
        forecast_type="solar",
        interval_minutes=15,
    )
    kp._model = cast(Prophet, DummyForecastModel())
    kp.config.efficiency_factor = 0.02
    kp.config.efficiency_profile = {6 * 60: 0.05}

    weather_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-06-01 06:00", "2024-06-01 12:00"], utc=True),
            "temperature_2m": [20.0, 25.0],
            "cloud_cover": [0.0, 0.0],
            "shortwave_radiation": [100.0, 100.0],
            "snow_depth": [0.0, 0.0],
            "snowfall": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(kp.weather_client, "fetch_forecast", lambda days: weather_df)
    monkeypatch.setattr(
        kp.weather_client, "resample_weather", lambda df, interval_minutes: df
    )

    forecast = kp.predict(days=1)

    assert forecast.loc[0, "yhat"] == pytest.approx(1.4375)
    assert forecast.loc[1, "yhat"] == pytest.approx(0.575)


def test_predict_does_not_apply_fixed_cloud_damping(monkeypatch, tmp_path) -> None:
    kp = KPowerForecast(
        model_id="test_no_cloud_damping",
        latitude=46.0,
        longitude=14.5,
        storage_path=str(tmp_path),
        forecast_type="solar",
        interval_minutes=15,
    )
    kp._model = cast(Prophet, DummyForecastModel())

    weather_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-06-01 12:00"], utc=True),
            "temperature_2m": [20.0],
            "cloud_cover": [100.0],
            "shortwave_radiation": [700.0],
            "snow_depth": [0.0],
            "snowfall": [0.0],
        }
    )
    monkeypatch.setattr(kp.weather_client, "fetch_forecast", lambda days: weather_df)
    monkeypatch.setattr(
        kp.weather_client, "resample_weather", lambda df, interval_minutes: df
    )

    forecast = kp.predict(days=1)

    assert forecast.loc[0, "pre_weather_correction_yhat"] == pytest.approx(10.0)
    assert forecast.loc[0, "weather_correction_multiplier"] == pytest.approx(1.0)
    assert forecast.loc[0, "yhat"] == pytest.approx(10.0)


def test_calibrate_weather_correction_uses_archive_weather_fallback(tmp_path) -> None:
    kp = KPowerForecast(
        model_id="test_weather_fallback",
        latitude=46.0,
        longitude=14.5,
        storage_path=str(tmp_path),
        forecast_type="solar",
        interval_minutes=15,
    )
    kp.config.efficiency_factor = 0.02

    timestamps = pd.date_range("2024-06-01 08:00", periods=12, freq="15min", tz="UTC")
    baseline_kwh = 100.0 * 0.02 * kp.config.efficiency_cap_headroom * 0.25
    df = pd.DataFrame(
        {
            "ds": timestamps,
            "y": [baseline_kwh * 1.2] * len(timestamps),
            "temperature_2m": [20.0] * len(timestamps),
            "cloud_cover": [90.0] * len(timestamps),
            "shortwave_radiation": [100.0] * len(timestamps),
            "snow_depth": [0.0] * len(timestamps),
            "snowfall": [0.0] * len(timestamps),
            "direct_radiation": [20.0] * len(timestamps),
            "diffuse_radiation": [80.0] * len(timestamps),
        }
    )

    correction = kp._calibrate_weather_correction(df)

    assert correction is not None
    assert correction["source"] == "historical_archive_weather"
    assert correction["default_multiplier"] == pytest.approx(1.2)


def test_predict_applies_optional_static_curtailment(monkeypatch, tmp_path) -> None:
    kp = KPowerForecast(
        model_id="test_curtailment",
        latitude=46.0,
        longitude=14.5,
        storage_path=str(tmp_path),
        forecast_type="solar",
        interval_minutes=15,
        inverter_ac_limit_kw=5.0,
    )
    kp._model = cast(Prophet, DummyForecastModel())

    weather_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-06-01 12:00"], utc=True),
            "temperature_2m": [20.0],
            "cloud_cover": [0.0],
            "shortwave_radiation": [700.0],
            "snow_depth": [0.0],
            "snowfall": [0.0],
        }
    )
    monkeypatch.setattr(kp.weather_client, "fetch_forecast", lambda days: weather_df)
    monkeypatch.setattr(
        kp.weather_client, "resample_weather", lambda df, interval_minutes: df
    )

    forecast = kp.predict(days=1)

    assert forecast.loc[0, "pre_curtailment_yhat"] == pytest.approx(10.0)
    assert forecast.loc[0, "applied_curtailment_limit_kwh"] == pytest.approx(1.25)
    assert forecast.loc[0, "curtailed_kwh"] == pytest.approx(8.75)
    assert forecast.loc[0, "curtailment_active"]
    assert forecast.loc[0, "yhat"] == pytest.approx(1.25)
