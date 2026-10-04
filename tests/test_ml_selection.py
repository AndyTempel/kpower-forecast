from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.ml import (
    KPowerMLConfig,
    KPowerMLForecast,
    MLBackendType,
    MLForecastType,
)
from kpower_forecast.ml.alignment import ForecastAlignmentError
from kpower_forecast.ml.baselines import BASELINE_NAME
from kpower_forecast.ml.selection import (
    BLEND_CANDIDATE,
    DEGREE_HOUR_CANDIDATE,
    ML_CANDIDATE,
    CandidateMetrics,
    DegreeHourRegression,
    score_windows,
    select_candidate,
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
    """Stub backend: a constant, or the true target when ``truth`` is set.

    Records each fit's last training timestamp and each prediction's first
    timestamp and feature columns.
    """

    minimum_contiguous_training_rows = 1

    def __init__(self, truth: pd.Series | None = None) -> None:
        self.truth = truth
        self.fit_ends: list[pd.Timestamp] = []
        self.predict_starts: list[pd.Timestamp] = []
        self.feature_columns: list[list[str]] = []

    def fit(self, history, features, calibration) -> None:
        self.fit_ends.append(pd.to_datetime(history["ds"], utc=True).max())
        self.feature_columns.append(list(features.columns))

    def predict(self, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        ds = pd.to_datetime(features["ds"], utc=True)
        self.predict_starts.append(ds.iloc[0])
        self.feature_columns.append(list(features.columns))
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
    assert set(metrics) == {
        ML_CANDIDATE,
        DEGREE_HOUR_CANDIDATE,
        BASELINE_NAME,
        BLEND_CANDIDATE,
    }
    assert metrics[ML_CANDIDATE]["origins"] >= 3
    assert metrics[DEGREE_HOUR_CANDIDATE]["rmse_1h"] < metrics[ML_CANDIDATE]["rmse_1h"]
    assert metrics[DEGREE_HOUR_CANDIDATE]["rmse_1h"] < metrics[BASELINE_NAME]["rmse_1h"]

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


def test_selection_keeps_ml_when_backtest_is_too_short(monkeypatch, tmp_path) -> None:
    # Seven days of history leave at most six complete day-ahead windows.
    weather = _weather("2026-01-05", _HOURS)
    history = pd.DataFrame({"ds": weather["ds"], "y": _heating(weather)})
    forecast = _forecast(
        monkeypatch,
        tmp_path,
        _Backend(),
        weather,
        selection_backtest_days=7,
        selection_min_origins=7,
    )

    forecast.train(history, force=True)

    assert forecast.selected_candidate == ML_CANDIDATE
    assert forecast.selection_reason == "backtest_too_short"
    assert forecast.candidate_metrics == {}


def test_backtest_trains_each_origin_only_on_earlier_rows(
    monkeypatch, tmp_path
) -> None:
    weather = _weather("2026-01-05", _HOURS)
    history = pd.DataFrame({"ds": weather["ds"], "y": _heating(weather)})
    backend = _Backend()
    forecast = _forecast(monkeypatch, tmp_path, backend, weather)

    forecast.train(history, force=True)

    interval = pd.Timedelta(hours=1)
    # The first fit is the calibration split and the last the full refit;
    # every backtest fit forecasts the slot right after its training rows.
    backtest = list(
        zip(backend.fit_ends[1:-1], backend.predict_starts[1:], strict=True)
    )
    assert len(backtest) >= 3
    for fit_end, predict_start in backtest:
        assert predict_start == fit_end + interval
        assert predict_start.tz_convert("Europe/Ljubljana").hour == 0


def test_artifact_from_other_selection_settings_is_not_restored(
    monkeypatch, tmp_path
) -> None:
    weather = _weather("2026-01-05", _HOURS)
    history = pd.DataFrame({"ds": weather["ds"], "y": _heating(weather)})
    _forecast(monkeypatch, tmp_path, _Backend(), weather).train(history, force=True)

    legacy = _forecast(
        monkeypatch, tmp_path, _Backend(), weather, candidate_selection=False
    )
    assert legacy.training_end is None

    # Changed regression settings must not reuse the stored winner either.
    retuned = _forecast(
        monkeypatch,
        tmp_path,
        _Backend(),
        weather,
        regression_base_temperature_c=18.0,
    )
    assert retuned.training_end is None
    same = _forecast(monkeypatch, tmp_path, _Backend(), weather)
    assert same.training_end is not None


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


def test_interval_calibration_uses_the_choice_made_without_each_origin(
    monkeypatch, tmp_path
) -> None:
    # ML is exact except for a disastrous first origin, so it loses overall.
    # Calibrating that origin with the overall winner would hide its error:
    # selected from the other origins alone, ML would have been served.
    weather = _weather("2026-01-05", _HOURS)
    forecast = _forecast(monkeypatch, tmp_path, _Backend(), weather)
    origins = list(pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC"))
    actual = np.full(24, 2.0)
    windows = {
        origin: {
            "actual": actual,
            ML_CANDIDATE: actual + (5.0 if index == 0 else 0.0),
            DEGREE_HOUR_CANDIDATE: actual + 0.5,
        }
        for index, origin in enumerate(origins)
    }
    names = [ML_CANDIDATE, DEGREE_HOUR_CANDIDATE]
    winner = forecast._select(forecast._score_backtest(windows, names, origins))

    choices = forecast._leave_one_out_choices(windows, names, origins, winner)

    assert winner == DEGREE_HOUR_CANDIDATE
    assert choices[origins[0]] == ML_CANDIDATE
    assert choices[origins[1]] == DEGREE_HOUR_CANDIDATE


def test_held_out_actuals_do_not_steer_their_own_blend_choice(
    monkeypatch, tmp_path
) -> None:
    # Blend weights fitted on the other origins must also leave out the
    # origin being calibrated, or its actuals pick its own candidate.
    weather = _weather("2026-01-05", _HOURS)
    forecast = _forecast(monkeypatch, tmp_path, _Backend(), weather)
    origins = list(pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC"))
    names = [ML_CANDIDATE, DEGREE_HOUR_CANDIDATE, BLEND_CANDIDATE]
    rng = np.random.default_rng(11)
    windows = {
        origin: {
            "actual": rng.uniform(0, 2, 24),
            ML_CANDIDATE: rng.uniform(0, 2, 24),
            DEGREE_HOUR_CANDIDATE: rng.uniform(0, 2, 24),
        }
        for origin in origins
    }
    changed = {origin: dict(window) for origin, window in windows.items()}
    changed[origins[0]]["actual"] = rng.uniform(0, 6, 24)

    def choice(source: dict) -> str:
        blended = forecast._with_blend(source, origins, held_out=None)
        return forecast._leave_one_out_choices(
            blended, names, origins, ML_CANDIDATE, blend=True
        )[origins[0]]

    assert choice(windows) == choice(changed)


def test_bias_guard_rejects_a_low_rmse_candidate_that_misses_energy() -> None:
    metrics = {
        ML_CANDIDATE: CandidateMetrics(
            rows=96, rmse=1.0, mae=0.5, bias=-0.05, rmse_1h=0.50, mean_actual=1.0
        ),
        DEGREE_HOUR_CANDIDATE: CandidateMetrics(
            rows=96, rmse=0.9, mae=0.5, bias=-0.30, rmse_1h=0.45, mean_actual=1.0
        ),
    }

    assert select_candidate(metrics) == DEGREE_HOUR_CANDIDATE
    assert select_candidate(metrics, bias_tolerance=0.15) == ML_CANDIDATE
    # When every candidate is outside the guard, the least biased wins.
    assert (
        select_candidate(
            {
                ML_CANDIDATE: metrics[ML_CANDIDATE],
                DEGREE_HOUR_CANDIDATE: metrics[DEGREE_HOUR_CANDIDATE],
            },
            bias_tolerance=0.01,
        )
        == ML_CANDIDATE
    )


def test_sliding_hour_rmse_forgives_a_run_shifted_by_one_slot() -> None:
    actual = np.zeros(96)
    actual[40:44] = 3.0
    near = np.roll(actual, 1)
    far = np.roll(actual, 8)

    near_score = score_windows([(actual, near)], window_steps=4)
    far_score = score_windows([(actual, far)], window_steps=4)

    assert near_score is not None and far_score is not None
    # Per slot the one-slot miss still costs half as much as the far miss;
    # per sliding hour it costs under a third.
    assert near_score.rmse == pytest.approx(0.5 * far_score.rmse)
    assert near_score.rmse_1h < 0.35 * far_score.rmse_1h


def test_blend_is_served_when_ml_and_regression_err_in_opposite_directions(
    monkeypatch, tmp_path
) -> None:
    # Day-alternating load the regression cannot see; the stub ML overshoots
    # it by as much as the regression undershoots, so an even blend is exact.
    weather = _weather("2026-01-05", _HOURS + 24)
    day = (np.arange(len(weather)) // 24) % 2
    swing = pd.Series(np.where(day == 0, 0.3, -0.3))
    target = _heating(weather) + 0.5 + swing
    ml = (target + swing).set_axis(weather["ds"])
    past = weather.iloc[:_HOURS]
    history = pd.DataFrame({"ds": past["ds"], "y": target.iloc[:_HOURS]})
    candidates = [ML_CANDIDATE, DEGREE_HOUR_CANDIDATE, BLEND_CANDIDATE]
    forecast = _forecast(
        monkeypatch, tmp_path, _Backend(ml), weather, selection_candidates=candidates
    )

    forecast.train(history, force=True)

    assert forecast.selected_candidate == BLEND_CANDIDATE
    assert forecast._blend_weight == pytest.approx(0.5, abs=0.1)
    restored = _forecast(
        monkeypatch, tmp_path, _Backend(ml), weather, selection_candidates=candidates
    )
    assert restored.selected_candidate == BLEND_CANDIDATE
    future = weather.iloc[_HOURS:].reset_index(drop=True)
    monkeypatch.setattr(
        restored,
        "_weather_for_model_grid",
        lambda *, start, horizon, forecast_days: future.iloc[:horizon],
    )
    result = restored.predict(days=1, origin=future["ds"].iloc[0].to_pydatetime())
    np.testing.assert_allclose(
        result["yhat"], target.iloc[_HOURS:].to_numpy(), atol=0.1
    )


@pytest.mark.parametrize("preserve_gaps", [True, False])
def test_known_covariates_reach_the_model_and_are_required_to_predict(
    monkeypatch, tmp_path, preserve_gaps: bool
) -> None:
    weather = _weather("2026-01-05", _HOURS + 24)
    past = weather.iloc[:_HOURS]
    mode = pd.Series((np.arange(len(weather)) % 24 >= 6).astype(float))
    history = pd.DataFrame(
        {
            "ds": past["ds"],
            "y": _heating(past),
            "hvac_mode": mode.iloc[:_HOURS].to_numpy(),
        }
    )
    backend = _Backend()
    forecast = _forecast(
        monkeypatch,
        tmp_path,
        backend,
        weather,
        candidate_selection=False,
        known_covariates=["hvac_mode"],
        preserve_gaps=preserve_gaps,
    )
    with pytest.raises(ValueError, match="known covariates"):
        forecast.train(history.drop(columns="hvac_mode"), force=True)

    forecast.train(history, force=True)

    assert all("hvac_mode" in columns for columns in backend.feature_columns)
    future = weather.iloc[_HOURS:].reset_index(drop=True)
    monkeypatch.setattr(
        forecast,
        "_weather_for_model_grid",
        lambda *, start, horizon, forecast_days: future.iloc[:horizon],
    )
    origin = future["ds"].iloc[0].to_pydatetime()
    with pytest.raises(ForecastAlignmentError, match="known_future"):
        forecast.predict(days=1, origin=origin)
    known = pd.DataFrame(
        {"ds": future["ds"], "hvac_mode": mode.iloc[_HOURS:].to_numpy()}
    )
    with pytest.raises(ForecastAlignmentError, match="every model-grid row"):
        forecast.predict(days=1, origin=origin, known_future=known.iloc[:-1])

    forecast.predict(days=1, origin=origin, known_future=known)

    assert "hvac_mode" in backend.feature_columns[-1]
    # A model trained without the covariate is not restored for one with it.
    assert (
        _forecast(
            monkeypatch,
            tmp_path,
            _Backend(),
            weather,
            candidate_selection=False,
            preserve_gaps=preserve_gaps,
        ).training_end
        is None
    )


def test_selection_candidates_must_include_ml_and_known_names() -> None:
    base: dict[str, Any] = {
        "model_id": "heating",
        "latitude": 46.0,
        "longitude": 14.0,
        "forecast_type": MLForecastType.HVAC,
        "candidate_selection": True,
    }
    with pytest.raises(ValueError, match="must include"):
        KPowerMLConfig(**base, selection_candidates=[DEGREE_HOUR_CANDIDATE])
    with pytest.raises(ValueError, match="unknown"):
        KPowerMLConfig(**base, selection_candidates=[ML_CANDIDATE, "prophet"])
    with pytest.raises(ValueError, match="multiple of interval"):
        KPowerMLConfig(**base, profile_smoothing_minutes=20)


def test_regression_inputs_reject_target_leakage_and_missing_columns() -> None:
    with pytest.raises(ValueError, match="must not include"):
        KPowerMLConfig(
            model_id="heating",
            latitude=46.0,
            longitude=14.0,
            forecast_type=MLForecastType.HVAC,
            candidate_selection=True,
            regression_extra_features=["y"],
        )

    weather = _weather("2026-01-05", _HOURS)
    regression = DegreeHourRegression(
        base_temperature_c=16.0,
        extra_features=("shortwave_radiation",),
        timezone="Europe/Ljubljana",
    )
    assert regression.fit(weather.assign(y=_heating(weather)))
    with pytest.raises(ValueError, match="shortwave_radiation"):
        regression.predict(weather.drop(columns="shortwave_radiation"))
