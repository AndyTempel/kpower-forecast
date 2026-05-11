"""Nixtla hybrid backend using StatsForecast and MLForecast."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from kpower_forecast.ml.config import KPowerMLConfig
from kpower_forecast.ml.dependencies import ensure_optional_dependencies

STATE_FILE = "state.json"
MODELS_FILE = "models.joblib"


def to_nixtla_frame(df: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    """Convert a KPower dataframe into Nixtla's long format.

    Args:
        df: Dataframe containing ``ds`` and ``y`` columns.
        unique_id: Stable model identifier.

    Returns:
        Dataframe with ``unique_id``, ``ds``, and ``y`` columns.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = {"ds", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    output = df[["ds", "y"]].copy()
    output.insert(0, "unique_id", unique_id)
    output["ds"] = pd.to_datetime(output["ds"], utc=True).dt.tz_localize(None)
    return output


class NixtlaHybridBackend:
    """StatsForecast structural model plus MLForecast residual model."""

    def __init__(self, config: KPowerMLConfig):
        self.config = config
        self._feature_columns: list[str] = []
        self._last_observed: float = 0.0
        self._fitted = False
        self._stats_model: Any = None
        self._residual_model: Any = None

    def fit(
        self,
        history: pd.DataFrame,
        features: pd.DataFrame,
        calibration: pd.DataFrame,
    ) -> None:
        """Fit structural and residual models.

        The adapter records schema information immediately and validates the
        optional backend dependencies before training.
        """
        if history.empty:
            raise ValueError("history must not be empty")

        self._feature_columns = [
            column for column in features.columns if column != "ds"
        ]
        self._last_observed = float(history["y"].iloc[-1])

        ensure_optional_dependencies(
            ("lightgbm", "mlforecast", "statsforecast"), "Nixtla hybrid backend"
        )
        from lightgbm import LGBMRegressor
        from mlforecast import MLForecast
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, SeasonalNaive

        nixtla_history = to_nixtla_frame(history, self.config.model_id)
        seasonal_length = 24 if self.config.interval_minutes == 60 else 96
        self._stats_model = StatsForecast(
            models=[SeasonalNaive(season_length=seasonal_length), AutoETS()],
            freq=f"{self.config.interval_minutes}min",
            n_jobs=1,
        )
        self._stats_model.fit(nixtla_history)

        residual_training = self._build_residual_training_frame(
            history, seasonal_length=seasonal_length
        )
        residual_training["unique_id"] = self.config.model_id
        residual_training = residual_training[["unique_id", "ds", "y"]]
        residual_training["ds"] = pd.to_datetime(
            residual_training["ds"], utc=True
        ).dt.tz_localize(None)

        lgbm_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbosity": -1,
            **self.config.backend_params.get("lightgbm", {}),
        }
        self._residual_model = MLForecast(
            models={"lgbm": LGBMRegressor(**lgbm_params)},
            freq=f"{self.config.interval_minutes}min",
            lags=[1, seasonal_length],
        )
        self._residual_model.fit(residual_training)
        self._fitted = True

    def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Predict future values with the fitted backend."""
        if not self._fitted:
            raise RuntimeError("backend has not been fitted")

        future = future_features.head(horizon).copy()
        future["ds"] = pd.to_datetime(future["ds"], utc=True)

        baseline = self._predict_stats_baseline(horizon=len(future))
        residual = self._predict_residual_adjustment(horizon=len(future))
        yhat = baseline.reset_index(drop=True) + residual.reset_index(drop=True)

        output = pd.DataFrame({"ds": future["ds"], "yhat": yhat})
        output["yhat"] = pd.to_numeric(output["yhat"], errors="coerce").fillna(
            self._last_observed
        )
        return output

    def _build_residual_training_frame(
        self, history: pd.DataFrame, seasonal_length: int
    ) -> pd.DataFrame:
        """Build a residual target frame from seasonal-naive baseline errors."""
        residual_training = history[["ds", "y"]].copy()
        fallback = residual_training["y"].expanding(min_periods=1).mean().shift(1)
        baseline = residual_training["y"].shift(seasonal_length).fillna(fallback)
        baseline = baseline.fillna(self._last_observed)
        residual_training["y"] = residual_training["y"] - baseline
        return residual_training

    def _predict_stats_baseline(self, horizon: int) -> pd.Series:
        """Predict the structural baseline component."""
        if self._stats_model is None:
            return pd.Series([self._last_observed] * horizon, dtype="float64")

        forecast = self._stats_model.predict(h=horizon)
        model_columns = [
            column for column in forecast.columns if column not in {"unique_id", "ds"}
        ]
        if not model_columns:
            return pd.Series([self._last_observed] * horizon, dtype="float64")
        baseline = pd.Series(
            pd.to_numeric(forecast[model_columns[0]].to_numpy(), errors="coerce"),
            dtype="float64",
        )
        return baseline.fillna(self._last_observed)

    def _predict_residual_adjustment(self, horizon: int) -> pd.Series:
        """Predict the residual adjustment component."""
        if self._residual_model is None:
            return pd.Series([0.0] * horizon, dtype="float64")

        forecast = self._residual_model.predict(h=horizon)
        model_columns = [
            column for column in forecast.columns if column not in {"unique_id", "ds"}
        ]
        if not model_columns:
            return pd.Series([0.0] * horizon, dtype="float64")
        residual = pd.Series(
            pd.to_numeric(forecast[model_columns[0]].to_numpy(), errors="coerce"),
            dtype="float64",
        )
        return residual.fillna(0.0)

    def save(self, path: Path) -> dict[str, str]:
        """Persist backend metadata.

        Full Nixtla binary persistence is intentionally deferred behind the
        manifest contract while the package-level API stabilizes.
        """
        ensure_optional_dependencies(("joblib",), "Nixtla hybrid persistence")
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        state_path = path / STATE_FILE
        models_path = path / MODELS_FILE
        state = {
            "feature_columns": self._feature_columns,
            "last_observed": self._last_observed,
            "fitted": self._fitted,
        }
        with state_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)
        joblib.dump(
            {"stats_model": self._stats_model, "residual_model": self._residual_model},
            models_path,
        )
        return {"state": STATE_FILE, "models": MODELS_FILE}

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
        self._fitted = bool(state.get("fitted", False))

        models_path = path / MODELS_FILE
        if models_path.exists():
            ensure_optional_dependencies(("joblib",), "Nixtla hybrid persistence")
            import joblib

            models = joblib.load(models_path)
            self._stats_model = models.get("stats_model")
            self._residual_model = models.get("residual_model")

    def feature_schema(self) -> list[str]:
        """Return the feature columns learned during training."""
        return list(self._feature_columns)
