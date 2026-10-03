import warnings

import pandas as pd
import pytest

from kpower_forecast.ml.alignment import ForecastAlignmentError
from kpower_forecast.ml.backends.nixtla import NixtlaHybridBackend
from kpower_forecast.ml.config import (
    HybridStructure,
    KPowerMLConfig,
    MLBackendType,
    MLForecastType,
)
from kpower_forecast.ml.forecast import _structure_matches


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


def test_nixtla_backend_trains_across_separate_observation_segments() -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=15,
            forecast_type=MLForecastType.CONSUMPTION,
        )
    )
    timestamps = pd.date_range(
        "2026-05-01", periods=24 * 30 * 4, freq="15min", tz="UTC"
    )
    history = (
        pd.DataFrame(
            {
                "ds": timestamps,
                "y": [float(1 + index % 96) for index in range(len(timestamps))],
            }
        )
        .drop(index=range(200, 205))
        .reset_index(drop=True)
    )
    features = pd.DataFrame(
        {"ds": history["ds"], "temperature_2m": [10.0] * len(history)}
    )

    backend.fit(history, features, history.tail(24))
    future = pd.DataFrame(
        {
            "ds": pd.date_range(
                timestamps[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min"
            ),
            "temperature_2m": [10.0] * 4,
        }
    )

    result = backend.predict(future, horizon=4)

    assert result["ds"].tolist() == future["ds"].tolist()
    assert result["yhat"].notna().all()


def test_nixtla_backend_excludes_segments_too_short_for_residual_lags(
    monkeypatch,
) -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=15,
            forecast_type=MLForecastType.CONSUMPTION,
        )
    )
    timestamps = pd.DatetimeIndex([])
    for start, periods in (
        ("2026-05-01T00:00:00Z", 40),
        ("2026-05-03T00:00:00Z", 20),
        ("2026-05-05T00:00:00Z", 200),
    ):
        timestamps = timestamps.append(
            pd.date_range(start, periods=periods, freq="15min")
        )
    history = pd.DataFrame(
        {
            "ds": timestamps,
            "y": [float(1 + index % 96) for index in range(len(timestamps))],
        },
        index=range(1000, 1000 + len(timestamps)),
    )
    features = pd.DataFrame(
        {
            "ds": history["ds"].to_numpy(),
            "temperature_2m": [10.0] * len(history),
        }
    )
    observed_feature_frames: list[pd.DataFrame] = []
    build_exogenous_frame = backend._build_exogenous_frame

    def record_exogenous_frame(
        selected_features: pd.DataFrame, unique_ids: pd.Series | None = None
    ) -> pd.DataFrame:
        observed_feature_frames.append(selected_features.copy())
        return build_exogenous_frame(selected_features, unique_ids)

    monkeypatch.setattr(backend, "_build_exogenous_frame", record_exogenous_frame)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.fit(history, features, history.tail(24))

    assert not any("dropped completely" in str(item.message) for item in caught)
    assert len(observed_feature_frames) == 1
    assert observed_feature_frames[0]["ds"].tolist() == history["ds"].iloc[60:].tolist()


def test_residual_fallback_stays_local_to_each_observation_segment() -> None:
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
            "ds": pd.to_datetime(
                [
                    "2026-05-01T00:00:00Z",
                    "2026-05-01T01:00:00Z",
                    "2026-05-02T00:00:00Z",
                    "2026-05-02T01:00:00Z",
                ]
            ),
            "y": [10.0, 12.0, 100.0, 102.0],
        }
    )
    backend._last_observed = 102.0

    residuals = backend._build_residual_training_frame(
        history, features=history[["ds"]], seasonal_length=24
    )

    assert residuals["y"].tolist() == [0.0, 2.0, 0.0, 2.0]


def test_nixtla_backend_runs_residual_on_exact_post_training_grid() -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=15,
            forecast_type=MLForecastType.CONSUMPTION,
        )
    )
    backend._fitted = True
    backend._last_train_ds = pd.Timestamp("2026-08-12T10:30:00Z")
    backend._last_observed = 0.2
    future = pd.DataFrame(
        {"ds": pd.date_range("2026-08-12T10:45:00Z", periods=4, freq="15min")}
    )

    class StatsModel:
        def predict(self, h: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "unique_id": ["consumption"] * h,
                    "ds": future["ds"].dt.tz_localize(None),
                    "SeasonalNaive": [0.2] * h,
                }
            )

    class ResidualModel:
        calls = 0

        def predict(self, h: int, X_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
            self.calls += 1
            assert ids == ["consumption"]
            return pd.DataFrame(
                {
                    "unique_id": ["consumption"] * h,
                    "ds": X_df["ds"],
                    "lgbm": [0.05] * h,
                }
            )

    residual = ResidualModel()
    backend._stats_model = StatsModel()
    backend._residual_model = residual

    result = backend.predict(future, horizon=4)

    assert residual.calls == 1
    assert result["ds"].tolist() == future["ds"].tolist()
    assert result["yhat"].tolist() == pytest.approx([0.25] * 4)


def test_nixtla_backend_rejects_shifted_residual_grid() -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=15,
            forecast_type=MLForecastType.CONSUMPTION,
        )
    )
    backend._last_train_ds = pd.Timestamp("2026-08-12T10:30:00Z")
    backend._residual_model = object()
    shifted = pd.DataFrame(
        {"ds": pd.date_range("2026-08-12T11:00:00Z", periods=4, freq="15min")}
    )

    with pytest.raises(ForecastAlignmentError, match="expected 2026-08-12T10:45"):
        backend._predict_residual_adjustment(4, shifted)


def test_nixtla_backend_does_not_silence_residual_prediction_error() -> None:
    backend = NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=15,
            forecast_type=MLForecastType.CONSUMPTION,
        )
    )
    backend._last_train_ds = pd.Timestamp("2026-08-12T10:30:00Z")

    class RejectingResidualModel:
        def predict(self, h: int, X_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
            raise ValueError("X_df does not match expected grid")

    backend._residual_model = RejectingResidualModel()
    future = pd.DataFrame(
        {"ds": pd.date_range("2026-08-12T10:45:00Z", periods=4, freq="15min")}
    )

    with pytest.raises(ForecastAlignmentError, match="rejected the aligned"):
        backend._predict_residual_adjustment(4, future)


def _profile_direct_backend(tmp_timezone: str = "UTC") -> NixtlaHybridBackend:
    return NixtlaHybridBackend(
        KPowerMLConfig(
            model_id="consumption",
            latitude=46.0,
            longitude=14.0,
            interval_minutes=15,
            forecast_type=MLForecastType.CONSUMPTION,
            timezone=tmp_timezone,
            hybrid_structure=HybridStructure.PROFILE_DIRECT,
        )
    )


def _calendar_features(ds: pd.Series) -> pd.DataFrame:
    hours = ds.dt.hour + ds.dt.minute / 60.0
    return pd.DataFrame(
        {"ds": ds, "hour": hours, "temperature_2m": 10.0 + (hours - 12).abs() / 2}
    )


def test_profile_direct_does_not_replay_a_one_off_event_into_future_days(
    tmp_path,
) -> None:
    pytest.importorskip("lightgbm")
    ds = pd.Series(pd.date_range("2026-09-01", periods=21 * 96, freq="15min", tz="UTC"))
    y = 0.25 + 0.1 * (ds.dt.hour.between(17, 21)).astype(float)
    # A heat-pump run at 03:00 on the last day only.
    spike = ds.dt.date.eq(ds.iloc[-1].date()) & ds.dt.hour.eq(3)
    y = y.where(~spike, 1.25)
    history = pd.DataFrame({"ds": ds, "y": y})
    backend = _profile_direct_backend()

    backend.fit(history, _calendar_features(ds), history.tail(96))
    future_ds = pd.Series(
        pd.date_range(
            ds.iloc[-1] + pd.Timedelta(minutes=15), periods=5 * 96, freq="15min"
        )
    )
    forecast = backend.predict(_calendar_features(future_ds), horizon=5 * 96)

    at_three = forecast.loc[future_ds.dt.hour.eq(3).to_numpy(), "yhat"]
    # Seasonal-naive would replay 1.25; the mean profile spreads it over 21 days.
    assert at_three.max() < 0.4
    assert forecast["yhat"].min() > 0.1
    # Tuesday-Friday share profile and features: identical days mean the
    # prediction does not drift with lead time (no recursion).
    daily = forecast["yhat"].to_numpy().reshape(5, 96)
    assert future_ds.iloc[0].dayofweek == 1
    assert abs(daily[0] - daily[3]).max() < 1e-9

    backend.save(tmp_path)
    restored = _profile_direct_backend()
    restored.load(tmp_path)
    pd.testing.assert_frame_equal(
        restored.predict(_calendar_features(future_ds), horizon=5 * 96), forecast
    )

    # A recursive backend must not serve profile_direct state.
    recursive = NixtlaHybridBackend(
        backend.config.model_copy(
            update={"hybrid_structure": HybridStructure.RECURSIVE_SEASONAL_NAIVE}
        )
    )
    recursive.load(tmp_path)
    assert recursive._fitted is False


@pytest.mark.parametrize(
    ("forecast_type", "backend", "message"),
    [
        (MLForecastType.SOLAR, MLBackendType.NIXTLA_HYBRID, "not for solar"),
        # Other backends would silently ignore the requested structure.
        (MLForecastType.CONSUMPTION, MLBackendType.NEURALFORECAST, "nixtla_hybrid"),
    ],
)
def test_profile_direct_structure_is_rejected_where_it_cannot_apply(
    forecast_type: MLForecastType, backend: MLBackendType, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KPowerMLConfig(
            model_id="model",
            latitude=46.0,
            longitude=14.0,
            forecast_type=forecast_type,
            backend=backend,
            hybrid_structure=HybridStructure.PROFILE_DIRECT,
        )


def test_artifact_from_another_hybrid_structure_is_not_reused() -> None:
    class Manifest:
        def __init__(self, metadata: dict) -> None:
            self.metadata = metadata

    direct = KPowerMLConfig(
        model_id="consumption",
        latitude=46.0,
        longitude=14.0,
        forecast_type=MLForecastType.CONSUMPTION,
        hybrid_structure=HybridStructure.PROFILE_DIRECT,
    )
    recursive = direct.model_copy(
        update={"hybrid_structure": HybridStructure.RECURSIVE_SEASONAL_NAIVE}
    )
    legacy = Manifest({})
    stored_direct = Manifest(
        {"hybrid_structure": "profile_direct", "profile_lookback_days": 28}
    )

    assert _structure_matches(legacy, recursive)
    assert not _structure_matches(legacy, direct)
    assert _structure_matches(stored_direct, direct)
    assert not _structure_matches(
        stored_direct, direct.model_copy(update={"profile_lookback_days": 14})
    )
