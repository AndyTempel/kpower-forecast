from typing import cast

import pandas as pd
import pytest

from kpower_forecast.ml import KPowerMLForecast, MLBackendType, MLForecastType
from kpower_forecast.ml.dependencies import (
    MissingMLDependencyError,
    ensure_optional_dependencies,
)


def test_ml_forecast_train_predict_with_neuralforecast_backend(
    monkeypatch, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id="ml-consumption",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "y": [1.0, 1.1, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2],
        }
    )
    weather = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            "temperature_2m": [10.0] * 24,
            "cloud_cover": [20.0] * 24,
            "shortwave_radiation": [0.0] * 24,
            "snow_depth": [0.0] * 24,
            "snowfall": [0.0] * 24,
        }
    )

    monkeypatch.setattr(
        forecast.weather_client, "fetch_historical", lambda start, end: weather
    )
    monkeypatch.setattr(forecast.weather_client, "fetch_forecast", lambda days: weather)
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )

    forecast.train(history, force=True)
    result = forecast.predict(days=1)

    assert len(result) == 24
    assert {"yhat", "yhat_lower_50", "yhat_upper_90"}.issubset(result.columns)
    assert forecast.storage.load_manifest() is not None

    restored = KPowerMLForecast(
        model_id="ml-consumption",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    monkeypatch.setattr(restored.weather_client, "fetch_forecast", lambda days: weather)
    monkeypatch.setattr(
        restored.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )

    restored_result = restored.predict(days=1)

    assert len(restored_result) == 24
    assert {"yhat", "yhat_lower_50", "yhat_upper_90"}.issubset(restored_result.columns)


def test_ml_forecast_applies_static_pv_inverter_curtailment(tmp_path) -> None:
    forecast = KPowerMLForecast(
        model_id="pv-static",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.SOLAR,
        backend=MLBackendType.NEURALFORECAST,
        inverter_ac_limit_kw=1.0,
        grid_export_limit_kw=0.6,
    )
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-01T10:00:00Z", periods=2, freq="30min"),
            "yhat": [2.0, 2.0],
            "yhat_lower_90": [1.5, 1.5],
            "yhat_upper_90": [2.5, 2.5],
        }
    )

    result = forecast._apply_pv_curtailment(frame)

    assert result[["yhat", "yhat_lower_90", "yhat_upper_90"]].max().max() == 0.6


def test_hvac_history_normalization_preserves_celsius_target(tmp_path) -> None:
    forecast = KPowerMLForecast(
        model_id="hvac-temp",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.HVAC,
        backend=MLBackendType.NEURALFORECAST,
    )
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC"),
            "y": [20.0, 20.5, 21.0, 21.5],
            "hvac_power_w": [0.0, 500.0, 1000.0, 1000.0],
        }
    )

    normalized = forecast._normalize_hvac_history(history)

    assert normalized["y"].tolist() == [20.25, 21.25]
    assert normalized["hvac_power_w"].tolist() == [250.0, 1000.0]


def test_ml_forecast_applies_dynamic_export_curtailment(tmp_path) -> None:
    forecast = KPowerMLForecast(
        model_id="pv-dynamic",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.SOLAR,
        backend=MLBackendType.NEURALFORECAST,
        inverter_ac_limit_kw=1.0,
    )
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-01T10:00:00Z", periods=2, freq="h"),
            "yhat": [2.0, 2.0],
            "yhat_upper_90": [3.0, 3.0],
        }
    )
    dynamic_limits = pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-01T10:00:00Z", periods=2, freq="h"),
            "export_limit_kw": [0.8, 0.4],
        }
    )

    result = forecast._apply_pv_curtailment(frame, dynamic_export_limits=dynamic_limits)

    assert result["yhat"].tolist() == [0.8, 0.4]
    assert result["yhat_upper_90"].tolist() == [0.8, 0.4]


def test_optional_dependency_boundary_reports_install_hint(monkeypatch) -> None:
    monkeypatch.setattr("kpower_forecast.ml.dependencies.find_spec", lambda name: None)

    with pytest.raises(MissingMLDependencyError, match=r"kpower-forecast\[ml\]"):
        ensure_optional_dependencies(("missing_backend",), "Missing backend")


def test_optional_dependency_boundary_supports_ai_extra_hint(monkeypatch) -> None:
    monkeypatch.setattr("kpower_forecast.ml.dependencies.find_spec", lambda name: None)

    with pytest.raises(MissingMLDependencyError, match=r"kpower-forecast\[ai\]"):
        ensure_optional_dependencies(
            ("neuralforecast",), "NeuralForecast backend", extra="ai"
        )
