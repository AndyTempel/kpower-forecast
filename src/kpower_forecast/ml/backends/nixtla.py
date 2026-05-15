"""Nixtla hybrid backend using StatsForecast and MLForecast."""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from kpower_forecast.ml.config import KPowerMLConfig, MLForecastType
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
        self._last_train_ds: pd.Timestamp | None = None
        self._solar_global_factor: float | None = None
        self._solar_profile: dict[int, float] = {}
        self._observed_history = pd.DataFrame(columns=["ds", "y"])
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

        self._feature_columns = self._select_feature_columns(features)
        self._last_observed = float(history["y"].iloc[-1])
        self._last_train_ds = pd.to_datetime(history["ds"], utc=True).max()
        self._observed_history = self._normalize_observed_history(history)

        ensure_optional_dependencies(
            ("lightgbm", "mlforecast", "statsforecast"), "Nixtla hybrid backend"
        )
        from lightgbm import LGBMRegressor
        from mlforecast import MLForecast
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, SeasonalNaive

        nixtla_history = to_nixtla_frame(history, self.config.model_id)
        seasonal_length = 24 if self.config.interval_minutes == 60 else 96
        self._fit_solar_profile(history=history, features=features)
        self._stats_model = StatsForecast(
            models=[SeasonalNaive(season_length=seasonal_length), AutoETS()],
            freq=f"{self.config.interval_minutes}min",
            n_jobs=1,
        )
        self._stats_model.fit(nixtla_history)

        residual_training = self._build_residual_training_frame(
            history, features=features, seasonal_length=seasonal_length
        )
        residual_training["unique_id"] = self.config.model_id
        residual_training = residual_training[["unique_id", "ds", "y"]]
        residual_training["ds"] = pd.to_datetime(
            residual_training["ds"], utc=True
        ).dt.tz_localize(None)
        residual_training = pd.merge(
            residual_training,
            self._build_exogenous_frame(features),
            on=["unique_id", "ds"],
            how="left",
        ).fillna(0.0)

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
        self._residual_model.fit(residual_training, static_features=[])
        self._fitted = True

    def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Predict future values with the fitted backend."""
        if not self._fitted:
            raise RuntimeError("backend has not been fitted")

        future = future_features.head(horizon).copy()
        future["ds"] = pd.to_datetime(future["ds"], utc=True)

        baseline = self._predict_stats_baseline(horizon=len(future))
        solar_baseline = self._predict_solar_baseline(future)
        if solar_baseline is not None:
            baseline = solar_baseline
        residual = self._predict_residual_adjustment(
            horizon=len(future), future_features=future_features
        )
        yhat = baseline.reset_index(drop=True) + residual.reset_index(drop=True)

        output = pd.DataFrame({"ds": future["ds"], "yhat": yhat})
        output["yhat"] = pd.to_numeric(output["yhat"], errors="coerce").fillna(
            self._last_observed
        )
        output = self._apply_observed_overlap(output)
        return output

    def _build_residual_training_frame(
        self, history: pd.DataFrame, features: pd.DataFrame, seasonal_length: int
    ) -> pd.DataFrame:
        """Build a residual target frame from seasonal-naive baseline errors."""
        residual_training = history[["ds", "y"]].copy()
        solar_baseline = self._predict_solar_baseline(features)
        if solar_baseline is None:
            fallback = residual_training["y"].expanding(min_periods=1).mean().shift(1)
            baseline = residual_training["y"].shift(seasonal_length).fillna(fallback)
            baseline = baseline.fillna(self._last_observed)
        else:
            baseline = solar_baseline.reindex(residual_training.index).fillna(0.0)
        residual_training["y"] = residual_training["y"] - baseline
        return residual_training

    def _fit_solar_profile(self, history: pd.DataFrame, features: pd.DataFrame) -> None:
        """Learn interval kWh per W/m2 by minute-of-day for solar forecasts."""
        if self.config.forecast_type != MLForecastType.SOLAR:
            return
        if "shortwave_radiation" not in features.columns:
            return

        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(history["ds"], utc=True),
                "y": pd.to_numeric(history["y"], errors="coerce"),
                "shortwave_radiation": pd.to_numeric(
                    features["shortwave_radiation"], errors="coerce"
                ),
            }
        ).dropna()
        frame = frame[(frame["shortwave_radiation"] > 20.0) & (frame["y"] >= 0.0)]
        if len(frame) < 12:
            return

        frame["factor"] = frame["y"] / frame["shortwave_radiation"]
        frame = frame[(frame["factor"] > 0.0) & frame["factor"].notna()]
        if len(frame) < 12:
            return

        upper = float(frame["factor"].quantile(0.98))
        frame = frame[frame["factor"] <= upper]
        self._solar_global_factor = float(frame["factor"].median())

        frame["minute_of_day"] = frame["ds"].dt.hour * 60 + frame["ds"].dt.minute
        grouped = frame.groupby("minute_of_day")["factor"].median()
        slots = pd.Index(range(0, 1440, self.config.interval_minutes))
        profile = grouped.reindex(slots).interpolate(limit_direction="both")
        profile = profile.fillna(self._solar_global_factor)
        profile = profile.rolling(window=3, center=True, min_periods=1).median()
        profile = profile.clip(lower=0.0, upper=upper)
        self._solar_profile = {
            int(cast(int, slot)): float(value)
            for slot, value in profile.dropna().items()
        }

    def _predict_solar_baseline(self, features: pd.DataFrame) -> pd.Series | None:
        """Predict a weather-aware solar baseline from shortwave radiation."""
        if (
            self._solar_global_factor is None
            or not self._solar_profile
            or "shortwave_radiation" not in features.columns
        ):
            return None

        frame = features[["ds", "shortwave_radiation"]].copy()
        frame["ds"] = pd.to_datetime(frame["ds"], utc=True)
        frame["shortwave_radiation"] = pd.to_numeric(
            frame["shortwave_radiation"], errors="coerce"
        ).fillna(0.0)
        minute_of_day = frame["ds"].dt.hour * 60 + frame["ds"].dt.minute
        factors = minute_of_day.map(
            lambda minute: self._solar_profile.get(
                int(minute), self._solar_global_factor or 0.0
            )
        )
        baseline = frame["shortwave_radiation"].clip(lower=0.0) * factors
        return pd.Series(
            pd.to_numeric(baseline, errors="coerce"), index=frame.index, dtype="float64"
        ).fillna(0.0)

    def _normalize_observed_history(self, history: pd.DataFrame) -> pd.DataFrame:
        """Normalize observed training targets for elapsed forecast intervals."""
        observed = history[["ds", "y"]].copy()
        observed["ds"] = pd.to_datetime(observed["ds"], utc=True)
        observed["y"] = pd.to_numeric(observed["y"], errors="coerce")
        return observed.dropna(subset=["ds", "y"]).drop_duplicates(
            subset=["ds"], keep="last"
        )

    def _apply_observed_overlap(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Replace elapsed forecast intervals with known observed production."""
        if self._observed_history.empty:
            return forecast

        output = forecast.copy()
        output["ds"] = pd.to_datetime(output["ds"], utc=True)
        observed = self._observed_history.rename(columns={"y": "observed_y"})
        output = pd.merge(output, observed, on="ds", how="left")
        observed_mask = output["observed_y"].notna()
        output.loc[observed_mask, "yhat"] = output.loc[observed_mask, "observed_y"]
        return output.drop(columns=["observed_y"])

    def _select_feature_columns(self, features: pd.DataFrame) -> list[str]:
        """Select numeric dynamic exogenous feature columns for residual learning."""
        feature_columns: list[str] = []
        for column in features.columns:
            if column in {"ds", "y"}:
                continue
            numeric = pd.to_numeric(features[column], errors="coerce")
            if numeric.notna().any():
                feature_columns.append(column)
        return feature_columns

    def _build_exogenous_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Build Nixtla-compatible dynamic exogenous feature frame."""
        frame = features[["ds"]].copy()
        for column in self._feature_columns:
            if column in features.columns:
                frame[column] = features[column]
            else:
                frame[column] = 0.0
        frame.insert(0, "unique_id", self.config.model_id)
        frame["ds"] = pd.to_datetime(frame["ds"], utc=True).dt.tz_localize(None)
        for column in self._feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return frame

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

    def _predict_residual_adjustment(
        self, horizon: int, future_features: pd.DataFrame
    ) -> pd.Series:
        """Predict the residual adjustment component."""
        if self._residual_model is None:
            return pd.Series([0.0] * horizon, dtype="float64")
        if self._last_train_ds is not None:
            future_start = pd.to_datetime(future_features["ds"].iloc[0], utc=True)
            expected_start = self._last_train_ds + pd.Timedelta(
                minutes=self.config.interval_minutes
            )
            if future_start != expected_start:
                return pd.Series([0.0] * horizon, dtype="float64")

        try:
            forecast = self._residual_model.predict(
                h=horizon,
                X_df=self._build_exogenous_frame(future_features),
            )
        except ValueError:
            return pd.Series([0.0] * horizon, dtype="float64")
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
            "last_train_ds": (
                None if self._last_train_ds is None else self._last_train_ds.isoformat()
            ),
            "solar_global_factor": self._solar_global_factor,
            "solar_profile": self._solar_profile,
            "fitted": self._fitted,
        }
        with state_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)
        joblib.dump(
            {
                "stats_model": self._stats_model,
                "residual_model": self._residual_model,
                "observed_history": self._observed_history,
            },
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
        last_train_ds = state.get("last_train_ds")
        self._last_train_ds = (
            None if last_train_ds is None else pd.to_datetime(last_train_ds, utc=True)
        )
        solar_global_factor = state.get("solar_global_factor")
        self._solar_global_factor = (
            None if solar_global_factor is None else float(solar_global_factor)
        )
        self._solar_profile = {
            int(key): float(value)
            for key, value in dict(state.get("solar_profile", {})).items()
        }
        self._fitted = bool(state.get("fitted", False))

        models_path = path / MODELS_FILE
        if models_path.exists():
            ensure_optional_dependencies(("joblib",), "Nixtla hybrid persistence")
            import joblib

            models = joblib.load(models_path)
            self._stats_model = models.get("stats_model")
            self._residual_model = models.get("residual_model")
            observed_history = models.get("observed_history")
            if isinstance(observed_history, pd.DataFrame):
                self._observed_history = self._normalize_observed_history(
                    observed_history
                )

    def feature_schema(self) -> list[str]:
        """Return the feature columns learned during training."""
        return list(self._feature_columns)
