import pandas as pd
import pytest

from kpower_forecast.ml.alignment import ForecastAlignmentError
from kpower_forecast.ml.backends.neural import NeuralForecastBackend
from kpower_forecast.ml.config import KPowerMLConfig, MLBackendType


class FixedHorizonModel:
    h = 4

    def fit(self, data: pd.DataFrame) -> None:
        return None

    def predict(self) -> pd.DataFrame:
        raise AssertionError("mismatched fixed-horizon model must not run")


def test_neural_backend_rejects_runtime_horizon_mismatch() -> None:
    backend = NeuralForecastBackend(
        KPowerMLConfig(
            model_id="neural",
            latitude=46.0,
            longitude=14.0,
            backend=MLBackendType.NEURALFORECAST,
        )
    )
    backend._fitted = True
    backend._last_train_ds = pd.Timestamp("2026-08-12T10:30:00Z")
    backend._configured_horizon = 4
    backend._model = FixedHorizonModel()
    future = pd.DataFrame(
        {"ds": pd.date_range("2026-08-12T10:45:00Z", periods=8, freq="15min")}
    )

    with pytest.raises(ForecastAlignmentError, match="fixed horizon 4; requested 8"):
        backend.predict(future, horizon=8)


def test_neural_backend_requires_one_positive_model_horizon() -> None:
    backend = NeuralForecastBackend(
        KPowerMLConfig(model_id="neural", latitude=46.0, longitude=14.0)
    )
    second = FixedHorizonModel()
    second.h = 8

    with pytest.raises(ValueError, match="same horizon"):
        backend._resolve_model_horizon([FixedHorizonModel(), second])
