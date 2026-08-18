from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from kpower_forecast.ml import (
    FORECAST_CONTRACT_VERSION,
    ForecastAlignmentError,
    KPowerMLForecast,
    MLBackendType,
    MLForecastType,
)
from kpower_forecast.ml.dependencies import (
    MissingMLDependencyError,
    ensure_optional_dependencies,
)
from kpower_forecast.ml.storage import MLModelManifest, MLModelStorage


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
            "ds": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
            "temperature_2m": [10.0] * 48,
            "cloud_cover": [20.0] * 48,
            "shortwave_radiation": [0.0] * 48,
            "snow_depth": [0.0] * 48,
            "snowfall": [0.0] * 48,
        }
    )

    monkeypatch.setattr(
        forecast.weather_client, "fetch_historical", lambda start, end: weather
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "fetch_forecast",
        lambda days, past_days=0: weather,
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )

    forecast.train(history, force=True)
    origin = datetime(2024, 1, 1, 8, tzinfo=timezone.utc)
    result = forecast.predict(days=1, origin=origin)

    assert len(result) == 24
    assert {"yhat", "yhat_lower_50", "yhat_upper_90"}.issubset(result.columns)
    manifest = forecast.storage.load_manifest()
    assert manifest is not None
    assert manifest.contract_version == FORECAST_CONTRACT_VERSION
    assert manifest.package_version == "2026.8.1"
    assert forecast.training_end == datetime(2024, 1, 1, 7, tzinfo=timezone.utc)

    restored = KPowerMLForecast(
        model_id="ml-consumption",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    monkeypatch.setattr(
        restored.weather_client,
        "fetch_forecast",
        lambda days, past_days=0: weather,
    )
    monkeypatch.setattr(
        restored.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )

    restored_result = restored.predict(days=1, origin=origin)
    baseline_result = restored.predict_baseline(
        days=1,
        origin=datetime(2024, 1, 1, 8, tzinfo=timezone.utc),
    )

    assert len(restored_result) == 24
    assert {"yhat", "yhat_lower_50", "yhat_upper_90"}.issubset(restored_result.columns)
    assert len(baseline_result) == 24
    assert baseline_result["ds"].iloc[0] == pd.Timestamp("2024-01-01T08:00:00Z")


def test_ml_forecast_calibrates_against_sanitized_predictions(
    monkeypatch, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id="sanitized-calibration",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )

    class NegativeCalibrationBackend:
        def fit(
            self,
            history: pd.DataFrame,
            features: pd.DataFrame,
            calibration: pd.DataFrame,
        ) -> None:
            return None

        def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "ds": future_features["ds"].iloc[:horizon],
                    "yhat": [-1.0] * horizon,
                }
            )

        def save(self, path: Path) -> dict[str, str]:
            return {}

        def feature_schema(self) -> list[str]:
            return []

    forecast.backend = cast(Any, NegativeCalibrationBackend())
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "y": [1.0] * 8,
        }
    )
    weather = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "temperature_2m": [10.0] * 8,
        }
    )
    monkeypatch.setattr(
        forecast.weather_client, "fetch_historical", lambda start, end: weather
    )

    forecast.train(history, force=True)

    assert set(forecast.conformal.quantiles.values()) == {1.0}


def test_ml_forecast_aligns_backend_grid_and_slices_explicit_origin(
    monkeypatch, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id="aligned-consumption",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=15,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    forecast._training_end = pd.Timestamp("2026-08-12T08:30:00Z")
    observed: dict[str, object] = {}

    class RecordingBackend:
        def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
            observed["first"] = future_features["ds"].iloc[0]
            observed["last"] = future_features["ds"].iloc[-1]
            observed["horizon"] = horizon
            points = [0.25] * horizon
            points[5] = -0.01
            return pd.DataFrame({"ds": future_features["ds"], "yhat": points})

        def fit(
            self,
            history: pd.DataFrame,
            features: pd.DataFrame,
            calibration: pd.DataFrame,
        ) -> None:
            return None

        def save(self, path: Path) -> dict[str, str]:
            return {}

        def load(self, path: Path) -> None:
            return None

        def feature_schema(self) -> list[str]:
            return []

    forecast.backend = cast(Any, RecordingBackend())
    weather = pd.DataFrame(
        {
            "ds": pd.date_range("2026-08-12T00:00:00Z", periods=288, freq="15min"),
            "temperature_2m": [20.0] * 288,
        }
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "fetch_forecast",
        lambda days, past_days=0: weather,
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )

    result = forecast.predict(
        days=1,
        origin=datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc),
    )

    assert observed == {
        "first": pd.Timestamp("2026-08-12T08:45:00Z"),
        "last": pd.Timestamp("2026-08-13T09:45:00Z"),
        "horizon": 101,
    }
    assert len(result) == 96
    assert result["yhat"].iloc[0] == 0.0
    assert result["ds"].iloc[0] == pd.Timestamp("2026-08-12T10:00:00Z")
    assert result["ds"].iloc[-1] == pd.Timestamp("2026-08-13T09:45:00Z")


@pytest.mark.parametrize(
    "forecast_type",
    [MLForecastType.CONSUMPTION, MLForecastType.HVAC],
)
def test_ml_forecast_clips_negative_energy_points(
    forecast_type: MLForecastType, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id=f"negative-{forecast_type.value}",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        forecast_type=forecast_type,
        backend=MLBackendType.NEURALFORECAST,
    )
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-08-19T14:15:00Z", periods=3, freq="15min"),
            "yhat": [0.2, -0.01, 0.3],
        }
    )

    result = forecast._sanitize_point_forecast(frame)

    assert result["yhat"].tolist() == [0.2, 0.0, 0.3]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_ml_forecast_rejects_non_finite_energy_points(
    bad_value: float, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id="non-finite-consumption",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    frame = pd.DataFrame(
        {
            "ds": [pd.Timestamp("2026-08-19T14:30:00Z")],
            "yhat": [bad_value],
        }
    )

    with pytest.raises(ForecastAlignmentError) as exc_info:
        forecast._sanitize_point_forecast(frame)

    message = str(exc_info.value)
    assert "index=0" in message
    assert "2026-08-19T14:30:00+00:00" in message
    assert repr(bad_value) in message


def test_ml_forecast_preserves_positive_energy_points(tmp_path) -> None:
    forecast = KPowerMLForecast(
        model_id="positive-solar",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        forecast_type=MLForecastType.SOLAR,
        backend=MLBackendType.NEURALFORECAST,
    )
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-08-19T14:15:00Z", periods=3, freq="15min"),
            "yhat": [0.0, 0.2, 0.3],
        }
    )

    result = forecast._sanitize_point_forecast(frame)

    assert result["yhat"].tolist() == [0.0, 0.2, 0.3]


def test_ml_forecast_defaults_to_current_slot_and_loads_elapsed_weather(
    monkeypatch, tmp_path
) -> None:
    interval_minutes = 15
    current_start = pd.Timestamp.now(tz="UTC").ceil(f"{interval_minutes}min")
    model_start = current_start - pd.Timedelta(days=1)
    forecast = KPowerMLForecast(
        model_id="current-consumption",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=interval_minutes,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    forecast._training_end = model_start - pd.Timedelta(minutes=interval_minutes)
    observed: dict[str, object] = {}

    class RecordingBackend:
        def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
            observed["first"] = future_features["ds"].iloc[0]
            observed["horizon"] = horizon
            return pd.DataFrame({"ds": future_features["ds"], "yhat": [0.2] * horizon})

    forecast.backend = cast(Any, RecordingBackend())
    future_weather = pd.DataFrame(
        {
            "ds": pd.date_range(
                model_start,
                current_start + pd.Timedelta(days=2),
                freq=f"{interval_minutes}min",
            ),
            "temperature_2m": 20.0,
        }
    )
    historical_calls: list[tuple[object, object]] = []
    forecast_calls: list[tuple[int, int]] = []

    def fetch_historical(start: object, end: object) -> pd.DataFrame:
        historical_calls.append((start, end))
        raise AssertionError("recent elapsed weather must not use the archive API")

    def fetch_forecast(days: int, past_days: int = 0) -> pd.DataFrame:
        forecast_calls.append((days, past_days))
        return future_weather

    monkeypatch.setattr(forecast.weather_client, "fetch_historical", fetch_historical)
    monkeypatch.setattr(forecast.weather_client, "fetch_forecast", fetch_forecast)
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, requested_interval: cast(pd.DataFrame, frame),
    )

    result = forecast.predict(days=1)

    assert not historical_calls
    assert forecast_calls and forecast_calls[0][1] >= 1
    assert observed == {"first": model_start, "horizon": 192}
    assert result["ds"].iloc[0] == current_start
    assert result["ds"].iloc[-1] == current_start + pd.Timedelta(hours=23, minutes=45)


def test_ml_forecast_caps_recent_weather_to_configured_one_day(
    monkeypatch, tmp_path
) -> None:
    current_start = pd.Timestamp.now(tz="UTC").ceil("15min")
    model_start = current_start - pd.Timedelta(days=3)
    forecast = KPowerMLForecast(
        model_id="bounded-recent-weather",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=15,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    assert forecast.weather_client.config.recent_forecast_past_days == 1
    recent_start = current_start.floor("D") - pd.Timedelta(days=1)
    archive = pd.DataFrame(
        {
            "ds": pd.date_range(
                model_start,
                recent_start - pd.Timedelta(minutes=15),
                freq="15min",
            ),
            "temperature_2m": 20.0,
        }
    )
    recent = pd.DataFrame(
        {
            "ds": pd.date_range(
                recent_start, current_start + pd.Timedelta(days=1), freq="15min"
            ),
            "temperature_2m": 20.0,
        }
    )
    forecast_calls: list[tuple[int, int]] = []
    archive_calls: list[tuple[object, object]] = []

    def fetch_forecast(days: int, past_days: int = 0) -> pd.DataFrame:
        forecast_calls.append((days, past_days))
        return recent

    def fetch_historical(start: object, end: object) -> pd.DataFrame:
        archive_calls.append((start, end))
        return archive

    monkeypatch.setattr(forecast.weather_client, "fetch_forecast", fetch_forecast)
    monkeypatch.setattr(forecast.weather_client, "fetch_historical", fetch_historical)
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, requested_interval: cast(pd.DataFrame, frame),
    )

    weather = forecast._weather_for_model_grid(
        start=model_start,
        horizon=4 * 24 * 3,
        forecast_days=2,
    )

    assert forecast_calls == [(2, 1)]
    assert archive_calls
    assert weather["ds"].min() == model_start


def test_ml_forecast_rejects_missing_weather_grid_timestamp(
    monkeypatch, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id="missing-weather",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=15,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    forecast._training_end = pd.Timestamp("2026-08-12T08:30:00Z")
    weather = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2026-08-12T00:00:00Z", periods=192, freq="15min"
            ).delete(36),
            "temperature_2m": [20.0] * 191,
        }
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "fetch_forecast",
        lambda days, past_days=0: weather,
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )

    with pytest.raises(ForecastAlignmentError, match="missing 1 required timestamps"):
        forecast.predict(
            days=1,
            origin=datetime(2026, 8, 12, 8, 45, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    "origin, message",
    [
        (datetime(2026, 8, 12, 10, 0), "timezone-aware"),
        (datetime(2026, 8, 12, 10, 1, tzinfo=timezone.utc), "not aligned"),
        (
            datetime(2026, 8, 12, 8, 30, tzinfo=timezone.utc),
            "precedes first post-training slot",
        ),
    ],
)
def test_ml_forecast_rejects_invalid_explicit_origin(
    origin: datetime, message: str, tmp_path
) -> None:
    forecast = KPowerMLForecast(
        model_id="invalid-origin",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=15,
        forecast_type=MLForecastType.CONSUMPTION,
        backend=MLBackendType.NEURALFORECAST,
    )
    forecast._training_end = pd.Timestamp("2026-08-12T08:30:00Z")

    with pytest.raises(ForecastAlignmentError, match=message):
        forecast.predict(days=1, origin=origin)


def test_ml_forecast_permits_forced_retrain_of_legacy_manifest(
    monkeypatch, tmp_path
) -> None:
    storage = MLModelStorage(str(tmp_path), "legacy")
    storage.save_manifest(
        MLModelManifest(
            model_id="legacy",
            backend_type=MLBackendType.NEURALFORECAST.value,
            target_type=MLForecastType.CONSUMPTION.value,
            interval_levels=[50, 80, 90],
            feature_columns=[],
        )
    )

    forecast = KPowerMLForecast(
        model_id="legacy",
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
            "y": [1.0] * 8,
        }
    )
    weather = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
            "temperature_2m": [10.0] * 48,
        }
    )
    monkeypatch.setattr(
        forecast.weather_client, "fetch_historical", lambda start, end: weather
    )

    assert forecast.training_end is None
    with pytest.raises(ForecastAlignmentError, match="requires a full retrain"):
        forecast.train(history, force=False)

    forecast.train(history, force=True)

    manifest = storage.load_manifest()
    assert manifest is not None
    assert manifest.contract_version == FORECAST_CONTRACT_VERSION


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
