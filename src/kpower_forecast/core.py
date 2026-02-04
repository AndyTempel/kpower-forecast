import logging
from typing import Literal

import pandas as pd
from prophet import Prophet
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import ModelStorage
from .utils import calculate_solar_elevation
from .weather_client import WeatherClient

logger = logging.getLogger(__name__)

class KPowerConfig(BaseModel):
    model_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    storage_path: str = "./data"
    interval_minutes: int = Field(15)
    forecast_type: Literal["solar", "consumption"] = "solar"
    heat_pump_mode: bool = False

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

    def train(self, history_df: pd.DataFrame, force: bool = False):
        """
        Trains the Prophet model using the provided history.
        """
        if not force and self.storage.load_model(self.config.model_id):
            logger.info(
                f"Model {self.config.model_id} already exists. "
                "Use force=True to retrain."
            )
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

        m = Prophet()
        
        if self.config.forecast_type == "solar":
            m.add_regressor("temperature_2m")
            m.add_regressor("cloud_cover")
            m.add_regressor("shortwave_radiation")
        elif self.config.forecast_type == "consumption":
            if self.config.heat_pump_mode:
                m.add_regressor("temperature_2m")
        
        logger.info(f"Training Prophet model for {self.config.forecast_type}...")
        m.fit(df)
        
        self.storage.save_model(m, self.config.model_id)

    def predict(self, days: int = 7) -> pd.DataFrame:
        """
        Generates forecast for the next 'days' days.
        """
        m = self.storage.load_model(self.config.model_id)
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
        
        forecast = m.predict(future)
        result = forecast[["ds", "yhat"]].copy()
        
        if self.config.forecast_type == "solar":
            logger.info("Applying night mask for solar forecast...")
            elevations = calculate_solar_elevation(
                self.config.latitude, self.config.longitude, result["ds"]
            )
            result.loc[elevations < 0, "yhat"] = 0
        
        result["yhat"] = result["yhat"].clip(lower=0)
        
        return result