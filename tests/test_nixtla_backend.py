import pandas as pd
import pytest

from kpower_forecast.ml.alignment import ForecastAlignmentError
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

        def predict(self, h: int, X_df: pd.DataFrame) -> pd.DataFrame:
            self.calls += 1
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
        def predict(self, h: int, X_df: pd.DataFrame) -> pd.DataFrame:
            raise ValueError("X_df does not match expected grid")

    backend._residual_model = RejectingResidualModel()
    future = pd.DataFrame(
        {"ds": pd.date_range("2026-08-12T10:45:00Z", periods=4, freq="15min")}
    )

    with pytest.raises(ForecastAlignmentError, match="rejected the aligned"):
        backend._predict_residual_adjustment(4, future)
