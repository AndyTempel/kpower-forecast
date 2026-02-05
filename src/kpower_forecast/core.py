import logging
from typing import List, Literal, cast

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import ModelStorage
from .utils import calculate_solar_elevation, get_clear_sky_ghi
from .weather_client import WeatherClient

logger = logging.getLogger(__name__)


class PredictionInterval(BaseModel):
    timestamp: pd.Timestamp
    expected_kwh: float
    lower_bound_kwh: float  # P10
    upper_bound_kwh: float  # P90

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KPowerConfig(BaseModel):
    model_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    storage_path: str = "./data"
    interval_minutes: int = Field(15)
    forecast_type: Literal["solar", "consumption"] = "solar"
    heat_pump_mode: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0

    @field_validator("interval_minutes")
    @classmethod
    def check_interval(cls, v: int) -> int:
        if v not in (15, 60):
            raise ValueError("interval_minutes must be 15 or 60")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KPowerForecast:
    def __init__(
        self,
        model_id: str,
        latitude: float,
        longitude: float,
        storage_path: str = "./data",
        interval_minutes: int = 15,
        forecast_type: Literal["solar", "consumption"] = "solar",
        heat_pump_mode: bool = False,
    ):
        self.config = KPowerConfig(
            model_id=model_id,
            latitude=latitude,
            longitude=longitude,
            storage_path=storage_path,
            interval_minutes=interval_minutes,
            forecast_type=forecast_type,
            heat_pump_mode=heat_pump_mode,
        )

        self.weather_client = WeatherClient(
            lat=self.config.latitude, lon=self.config.longitude
        )
        self.storage = ModelStorage(storage_path=self.config.storage_path)
        self._model = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-informed features and rolling windows.
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"], utc=True)

        # 1. Physics: Clear Sky GHI
        logger.info("Calculating physics-informed Clear Sky GHI...")
        # Ensure index is DatetimeIndex for pvlib
        temp_df = df.set_index("ds")
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        df["clear_sky_ghi"] = get_clear_sky_ghi(
            self.config.latitude, self.config.longitude, temp_df.index
        ).values

        # 2. Rolling Cloud Cover (3-hour window)
        # 3 hours = 180 minutes. Window depends on interval_minutes.
        window_size = 180 // self.config.interval_minutes
        logger.info(f"Adding rolling cloud cover (window={window_size})...")
        df["rolling_cloud_cover"] = (
            df["cloud_cover"].rolling(window=window_size, min_periods=1).mean()
        )

        return df

    def train(self, history_df: pd.DataFrame, force: bool = False):
        """
        Trains the Prophet model using the provided history.
        """
        if not force:
            if self._model:
                logger.info(
                    f"Model {self.config.model_id} already exists (cached). "
                    "Use force=True to retrain."
                )
                return

            loaded_model = self.storage.load_model(self.config.model_id)
            if loaded_model:
                logger.info(
                    f"Model {self.config.model_id} already exists. "
                    "Use force=True to retrain."
                )
                self._model = loaded_model
                return

        df = history_df.copy()
        if "ds" not in df.columns or "y" not in df.columns:
            raise ValueError("history_df must contain 'ds' and 'y' columns")

        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        df = df.sort_values("ds")

        start_date = df["ds"].min().date()
        end_date = df["ds"].max().date()

        weather_df = self.weather_client.fetch_historical(start_date, end_date)
        weather_df = self.weather_client.resample_weather(
            weather_df, self.config.interval_minutes
        )

        df = pd.merge(df, weather_df, on="ds", how="left")

        weather_cols = ["temperature_2m", "cloud_cover", "shortwave_radiation"]
        df[weather_cols] = df[weather_cols].interpolate(method="linear").bfill().ffill()

        if df[weather_cols].isnull().any().any():
            df = df.dropna(subset=weather_cols)

        # Feature Engineering
        df = self._prepare_features(df)

        # Initialize Prophet with tuned hyperparameters
        m = Prophet(
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=0.8,  # Used for P10/P90 (80% interval)
        )

        if self.config.forecast_type == "solar":
            m.add_regressor("temperature_2m")
            m.add_regressor("rolling_cloud_cover")
            m.add_regressor("shortwave_radiation")
            m.add_regressor("clear_sky_ghi")
        elif self.config.forecast_type == "consumption":
            if self.config.heat_pump_mode:
                m.add_regressor("temperature_2m")

        logger.info(f"Training Prophet model for {self.config.forecast_type}...")
        # Prophet requires tz-naive datetimes
        df_prophet = df.copy()
        df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
        m.fit(df_prophet)

        self.storage.save_model(m, self.config.model_id)
        self._model = m

    def tune_model(self, history_df: pd.DataFrame, days: int = 30):
        """
        Find optimal hyperparameters using cross-validation.
        """
        logger.info(f"Tuning model hyperparameters using {days} days of history...")

        # We need to prepare data first as cross_validation needs the regressors
        # This is a bit complex as we need weather data for history_df
        # For simplicity, we assume train() logic but without fitting.

        # Prepare data (duplicated logic from train, could be refactored)
        df = history_df.copy()
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        start_date = df["ds"].min().date()
        end_date = df["ds"].max().date()
        weather_df = self.weather_client.fetch_historical(start_date, end_date)
        weather_df = self.weather_client.resample_weather(
            weather_df, self.config.interval_minutes
        )
        df = pd.merge(df, weather_df, on="ds", how="left")
        weather_cols = ["temperature_2m", "cloud_cover", "shortwave_radiation"]
        df[weather_cols] = df[weather_cols].interpolate(method="linear").bfill().ffill()
        df = self._prepare_features(df.dropna(subset=weather_cols))

        param_grid = {
            "changepoint_prior_scale": [0.001, 0.05, 0.5],
            "seasonality_prior_scale": [0.01, 1.0, 10.0],
        }

        # Simplified tuning loop
        best_params = {}
        min_rmse = float("inf")

        for cps in param_grid["changepoint_prior_scale"]:
            for sps in param_grid["seasonality_prior_scale"]:
                m = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
                if self.config.forecast_type == "solar":
                    m.add_regressor("temperature_2m")
                    m.add_regressor("rolling_cloud_cover")
                    m.add_regressor("shortwave_radiation")
                    m.add_regressor("clear_sky_ghi")
                elif (
                    self.config.forecast_type == "consumption"
                    and self.config.heat_pump_mode
                ):
                    m.add_regressor("temperature_2m")

                # Prophet requires tz-naive datetimes
                df_prophet = df.copy()
                df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
                m.fit(df_prophet)

                # Cross-validation
                # initial should be at least 3x horizon
                df_cv = cross_validation(
                    m, initial=f"{days // 2} days", period="5 days", horizon="5 days"
                )
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmse = df_p["rmse"].values[0]

                if rmse < min_rmse:
                    min_rmse = rmse
                    best_params = {"cps": cps, "sps": sps}

        logger.info(f"Best params found: {best_params} with RMSE {min_rmse}")
        self.config.changepoint_prior_scale = best_params["cps"]
        self.config.seasonality_prior_scale = best_params["sps"]

    def predict(self, days: int = 7) -> pd.DataFrame:
        """
        Generates forecast for the next 'days' days.
        """
        if self._model is None:
            self._model = self.storage.load_model(self.config.model_id)

        m = self._model
        if m is None:
            raise RuntimeError(
                f"Model {self.config.model_id} not found. Please run train() first."
            )

        weather_forecast = self.weather_client.fetch_forecast(days=days)
        weather_forecast = self.weather_client.resample_weather(
            weather_forecast, self.config.interval_minutes
        )

        future = pd.DataFrame({"ds": weather_forecast["ds"]})
        future = pd.merge(future, weather_forecast, on="ds", how="left")

        weather_cols = ["temperature_2m", "cloud_cover", "shortwave_radiation"]
        future[weather_cols] = (
            future[weather_cols].interpolate(method="linear").bfill().ffill()
        )

        # Feature Engineering
        future = self._prepare_features(future)

        # Prophet requires tz-naive datetimes
        future_prophet = future.copy()
        future_prophet["ds"] = future_prophet["ds"].dt.tz_localize(None)

        forecast = m.predict(future_prophet)

        # Night Mask & Clipping
        if self.config.forecast_type == "solar":
            logger.info("Applying night mask for solar forecast...")
            elevations = calculate_solar_elevation(
                self.config.latitude, self.config.longitude, forecast["ds"]
            )
            forecast.loc[elevations < 0, ["yhat", "yhat_lower", "yhat_upper"]] = 0

        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(lower=0)

        return cast(pd.DataFrame, forecast)

    def get_prediction_intervals(self, days: int = 7) -> List[PredictionInterval]:
        """
        Returns prediction intervals for EMS.
        """
        forecast = self.predict(days=days)

        intervals = []
        for _, row in forecast.iterrows():
            intervals.append(
                PredictionInterval(
                    timestamp=row["ds"],
                    expected_kwh=row["yhat"],
                    lower_bound_kwh=row["yhat_lower"],
                    upper_bound_kwh=row["yhat_upper"],
                )
            )
        return intervals

    def get_surplus_probability(
        self, threshold_kwh: float, days: int = 7
    ) -> pd.DataFrame:
        """
        Returns probability of exceeding threshold_kwh.
        Prophet doesn't provide direct probabilities, but we can estimate
        from the uncertainty interval (yhat_upper - yhat_lower)
        assuming normal distribution.
        """
        forecast = self.predict(days=days)

        # Estimate sigma from 80% interval (approx 1.28 * sigma)
        sigma = (forecast["yhat_upper"] - forecast["yhat_lower"]) / (2 * 1.28)
        sigma = sigma.replace(0, 1e-9)  # Avoid div by zero

        from scipy.stats import norm

        z_score = (threshold_kwh - forecast["yhat"]) / sigma
        prob_exceed = 1 - norm.cdf(z_score)

        return pd.DataFrame({"ds": forecast["ds"], "surplus_prob": prob_exceed})
