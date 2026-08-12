"""Nixtla NeuralForecast backend adapter."""

import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from kpower_forecast.ml.alignment import (
    ForecastAlignmentError,
    validate_timestamp_grid,
)
from kpower_forecast.ml.config import KPowerMLConfig
from kpower_forecast.ml.dependencies import ensure_optional_dependencies

from .nixtla import to_nixtla_frame

STATE_FILE = "state.json"
MODEL_FILE = "neuralforecast.joblib"


class NeuralForecastModel(Protocol):
    """Protocol for fitted NeuralForecast model instances."""

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model on Nixtla long-format data."""

    def predict(self) -> pd.DataFrame:
        """Predict values for the configured horizon."""


class NeuralForecastBackend:
    """Adapter for Nixtla NeuralForecast models.

    The default model selection is intentionally conservative. Callers can pass
    explicit NeuralForecast model objects via ``backend_params['models']`` once
    the optional dependencies are installed.
    """

    def __init__(self, config: KPowerMLConfig):
        self.config = config
        self._feature_columns: list[str] = []
        self._last_observed: float = 0.0
        self._last_train_ds: pd.Timestamp | None = None
        self._configured_horizon: int | None = None
        self._model: NeuralForecastModel | None = None
        self._fitted = False

    def fit(
        self,
        history: pd.DataFrame,
        features: pd.DataFrame,
        calibration: pd.DataFrame,
    ) -> None:
        """Fit configured NeuralForecast models."""
        if history.empty:
            raise ValueError("history must not be empty")

        self._feature_columns = [
            column for column in features.columns if column != "ds"
        ]
        self._last_observed = float(history["y"].iloc[-1])
        self._last_train_ds = pd.to_datetime(history["ds"], utc=True).max()
        models = self.config.backend_params.get("models")
        if not models:
            self._fitted = True
            return
        self._configured_horizon = self._resolve_model_horizon(models)

        ensure_optional_dependencies(
            ("neuralforecast",), "NeuralForecast backend", extra="ai"
        )
        from neuralforecast import NeuralForecast

        nixtla_history = to_nixtla_frame(history, self.config.model_id)
        self._model = NeuralForecast(
            models=models,
            freq=f"{self.config.interval_minutes}min",
        )
        self._model.fit(nixtla_history)
        self._fitted = True

    def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Predict future values with NeuralForecast when configured."""
        if not self._fitted:
            raise RuntimeError("backend has not been fitted")

        future = future_features.head(horizon).copy()
        future["ds"] = pd.to_datetime(future["ds"], utc=True)
        if self._last_train_ds is None:
            raise ForecastAlignmentError(
                "NeuralForecast training cutoff is unavailable"
            )
        expected_start = self._last_train_ds + pd.Timedelta(
            minutes=self.config.interval_minutes
        )
        validate_timestamp_grid(
            future,
            interval_minutes=self.config.interval_minutes,
            expected_start=expected_start,
            expected_length=horizon,
            label="NeuralForecast future features",
        )
        if self._model is None:
            yhat_values = pd.Series(
                [self._last_observed] * len(future), dtype="float64"
            )
        else:
            if self._configured_horizon != horizon:
                raise ForecastAlignmentError(
                    "NeuralForecast models use fixed horizon "
                    f"{self._configured_horizon}; requested {horizon}. "
                    "Retrain models with the requested horizon or omit custom models."
                )
            forecast = self._model.predict().reset_index()
            validate_timestamp_grid(
                forecast,
                interval_minutes=self.config.interval_minutes,
                expected_start=expected_start,
                expected_length=horizon,
                label="NeuralForecast output",
            )
            model_columns = [
                column
                for column in forecast.columns
                if column not in {"unique_id", "ds"}
            ]
            yhat_values = pd.Series(
                forecast[model_columns[0]].head(len(future)).to_numpy(),
                dtype="float64",
            )

        return pd.DataFrame({"ds": future["ds"], "yhat": yhat_values.to_numpy()})

    def save(self, path: Path) -> dict[str, str]:
        """Persist backend metadata."""
        artifacts = {"state": STATE_FILE}
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_columns": self._feature_columns,
            "last_observed": self._last_observed,
            "last_train_ds": (
                None if self._last_train_ds is None else self._last_train_ds.isoformat()
            ),
            "configured_horizon": self._configured_horizon,
            "fitted": self._fitted,
        }
        with (path / STATE_FILE).open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)

        if self._model is not None:
            ensure_optional_dependencies(
                ("joblib",), "NeuralForecast persistence", extra="ai"
            )
            import joblib

            joblib.dump(self._model, path / MODEL_FILE)
            artifacts["model"] = MODEL_FILE
        return artifacts

    def load(self, path: Path) -> None:
        """Load backend artifacts from ``path``."""
        state_path = path / STATE_FILE
        if not state_path.exists():
            self._fitted = False
            return

        with state_path.open(encoding="utf-8") as file:
            state = json.load(file)
        self._feature_columns = list(state.get("feature_columns", []))
        self._last_observed = float(state.get("last_observed", 0.0))
        last_train_ds = state.get("last_train_ds")
        self._last_train_ds = (
            None if last_train_ds is None else pd.to_datetime(last_train_ds, utc=True)
        )
        configured_horizon = state.get("configured_horizon")
        self._configured_horizon = (
            None if configured_horizon is None else int(configured_horizon)
        )
        self._fitted = bool(state.get("fitted", False))

        model_path = path / MODEL_FILE
        if model_path.exists():
            ensure_optional_dependencies(
                ("joblib",), "NeuralForecast persistence", extra="ai"
            )
            import joblib

            loaded_model = joblib.load(model_path)
            if not hasattr(loaded_model, "fit") or not hasattr(loaded_model, "predict"):
                raise RuntimeError("persisted NeuralForecast model is invalid")
            self._model = loaded_model

    def _resolve_model_horizon(self, models: object) -> int:
        """Return the common positive fixed horizon for configured models."""
        if not isinstance(models, (list, tuple)) or not models:
            raise ValueError("backend_params['models'] must be a non-empty sequence")
        horizons = {getattr(model, "h", None) for model in models}
        if len(horizons) != 1:
            raise ValueError("all NeuralForecast models must use the same horizon")
        horizon = next(iter(horizons))
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("NeuralForecast models must expose a positive 'h' horizon")
        return horizon

    def feature_schema(self) -> list[str]:
        """Return feature columns learned during training."""
        return list(self._feature_columns)
