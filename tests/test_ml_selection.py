from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.ml import KPowerMLForecast, MLBackendType, MLForecastType
from kpower_forecast.ml.baselines import BASELINE_NAME
from kpower_forecast.ml.selection import (
    DEGREE_HOUR_CANDIDATE,
    ML_CANDIDATE,
    DegreeHourRegression,
)

_HOURS = 168


def _weather(start: str, hours: int) -> pd.DataFrame:
    steps = np.arange(hours)
    # Daily cycle plus a multi-day swing, so degree hours vary within each
    # local hour and the regression is identifiable.
    temperature = (
        9.0
        + 4.0 * np.sin(steps * 2 * np.pi / 24)
        + 5.0 * np.sin(steps * 2 * np.pi / (24 * 4.3))
    )
    return pd.DataFrame(
        {
            "ds": pd.date_range(start, periods=hours, freq="h", tz="UTC"),
            "temperature_2m": temperature,
            "shortwave_radiation": 0.0,
        }
    )


def _heating(weather: pd.DataFrame) -> pd.Series:
    return 0.2 * np.clip(16.0 - weather["temperature_2m"], 0.0, None)


class _Backend:
    """Stub backend: a constant, or the true target when ``truth`` is set."""

    minimum_contiguous_training_rows = 1

    def __init__(self, truth: pd.Series | None = None) -> None:
        self.truth = truth

    def fit(self, history, features, calibration) -> None:
        return None

    def predict(self, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        ds = pd.to_datetime(features["ds"], utc=True)
        if self.truth is None:
            values = np.full(horizon, 1.0)
        else:
            values = self.truth.reindex(ds).to_numpy()
        return pd.DataFrame({"ds": ds.reset_index(drop=True), "yhat": values})

    def feature_schema(self) -> list[str]:
        return []

    def save(self, artifact_dir) -> dict[str, str]:
        return {}

    def load(self, artifact_dir) -> None:
        return None


def _forecast(
    monkeypatch, tmp_path, backend: _Backend, weather: pd.DataFrame, **overrides: Any
) -> KPowerMLForecast:
    monkeypatch.setattr(
        "kpower_forecast.ml.forecast.create_backend", lambda config: backend
    )
    settings: dict[str, Any] = {
        "preserve_gaps": True,
        "candidate_selection": True,
        "min_selection_holdout_rows": 24,
        "calibration_fraction": 0.3,
    }
    settings.update(overrides)
    forecast = KPowerMLForecast(
        model_id="heating",
        latitude=46.0,
        longitude=14.0,
        storage_path=str(tmp_path),
        interval_minutes=60,
        forecast_type=MLForecastType.HVAC,
        backend=MLBackendType.NEURALFORECAST,
        timezone="Europe/Ljubljana",
        **settings,
    )
    monkeypatch.setattr(
        forecast.weather_client, "fetch_historical", lambda start, end: weather
    )
    monkeypatch.setattr(
        forecast.weather_client,
        "resample_weather",
        lambda frame, interval_minutes: cast(pd.DataFrame, frame),
    )
    return forecast


def test_selection_serves_degree_hour_regression_when_it_beats_ml(
    monkeypatch, tmp_path
) -> None:
    weather = _weather("2026-01-05", _HOURS + 24)
    past = weather.iloc[:_HOURS]
    history = pd.DataFrame({"ds": past["ds"], "y": _heating(past)})
    forecast = _forecast(monkeypatch, tmp_path, _Backend(), weather)

    forecast.train(history, force=True)

    assert forecast.selected_candidate == DEGREE_HOUR_CANDIDATE
    metrics = forecast.candidate_metrics
    assert set(metrics) == {ML_CANDIDATE, DEGREE_HOUR_CANDIDATE, BASELINE_NAME}
    assert metrics[DEGREE_HOUR_CANDIDATE]["rmse"] < metrics[ML_CANDIDATE]["rmse"]
    assert metrics[DEGREE_HOUR_CANDIDATE]["rmse"] < metrics[BASELINE_NAME]["rmse"]

    # A reloaded instance restores the winner and serves it from weather.
    restored = _forecast(monkeypatch, tmp_path, _Backend(), weather)
    assert restored.selected_candidate == DEGREE_HOUR_CANDIDATE
    future = weather.iloc[_HOURS:].reset_index(drop=True)
    monkeypatch.setattr(
        restored,
        "_weather_for_model_grid",
        lambda *, start, horizon, forecast_days: future.iloc[:horizon],
    )

    result = restored.predict(days=1, origin=future["ds"].iloc[0].to_pydatetime())

    np.testing.assert_allclose(result["yhat"], _heating(future), atol=0.05)
    assert result["yhat_lower_90"].le(result["yhat"]).all()


def test_selection_keeps_ml_when_it_is_best(monkeypatch, tmp_path) -> None:
    weather = _weather("2026-01-05", _HOURS)
    truth = _heating(weather).set_axis(weather["ds"])
    history = pd.DataFrame({"ds": weather["ds"], "y": truth.to_numpy()})
    forecast = _forecast(monkeypatch, tmp_path, _Backend(truth), weather)

    forecast.train(history, force=True)

    assert forecast.selected_candidate == ML_CANDIDATE
    assert forecast.candidate_metrics[ML_CANDIDATE]["rmse"] == pytest.approx(0.0)
    manifest = forecast.storage.load_manifest()
    assert manifest is not None
    assert manifest.metadata["candidate_selection"]["selected"] == ML_CANDIDATE
    assert manifest.metadata["degree_hour_regression"] is None


def test_selection_keeps_ml_when_holdout_is_too_short(monkeypatch, tmp_path) -> None:
    weather = _weather("2026-01-05", _HOURS)
    history = pd.DataFrame({"ds": weather["ds"], "y": _heating(weather)})
    forecast = _forecast(
        monkeypatch, tmp_path, _Backend(), weather, min_selection_holdout_rows=500
    )

    forecast.train(history, force=True)

    assert forecast.selected_candidate == ML_CANDIDATE
    assert forecast.selection_reason == "holdout_too_short"
    assert forecast.candidate_metrics == {}


def test_artifact_from_other_selection_mode_is_not_restored(
    monkeypatch, tmp_path
) -> None:
    weather = _weather("2026-01-05", _HOURS)
    history = pd.DataFrame({"ds": weather["ds"], "y": _heating(weather)})
    _forecast(monkeypatch, tmp_path, _Backend(), weather).train(history, force=True)

    legacy = _forecast(
        monkeypatch, tmp_path, _Backend(), weather, candidate_selection=False
    )

    assert legacy.training_end is None


def test_degree_hour_regression_fits_when_a_local_hour_is_never_observed() -> None:
    weather = _weather("2026-01-05", _HOURS)
    frame = weather.assign(y=_heating(weather))
    local_hour = frame["ds"].dt.tz_convert("Europe/Ljubljana").dt.hour
    frame.loc[local_hour == 3, "y"] = float("nan")
    regression = DegreeHourRegression(
        base_temperature_c=16.0,
        extra_features=("shortwave_radiation",),
        timezone="Europe/Ljubljana",
    )

    assert regression.fit(frame)
    assert regression.predict(weather).notna().all()
