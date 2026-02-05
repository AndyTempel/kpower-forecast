import datetime
import logging
from typing import Optional

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WeatherConfig(BaseModel):
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"


class WeatherClient:
    def __init__(self, lat: float, lon: float, config: Optional[WeatherConfig] = None):
        self.lat = lat
        self.lon = lon
        self.config = config or WeatherConfig()

    def fetch_historical(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for training.
        """
        params: dict[str, str | float | list[str]] = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": [
                "temperature_2m",
                "cloud_cover",
                "shortwave_radiation",
                "snow_depth",
                "snowfall",
            ],
            "timezone": "UTC",
        }

        try:
            logger.info(f"Fetching historical weather from {self.config.archive_url}")
            response = requests.get(self.config.archive_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._process_response(data)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch historical weather: {e}")
            raise

    def fetch_forecast(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch weather forecast for prediction.
        """
        params: dict[str, str | float | int | list[str]] = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": [
                "temperature_2m",
                "cloud_cover",
                "shortwave_radiation",
                "snow_depth",
                "snowfall",
            ],
            "forecast_days": days,
            "timezone": "UTC",
        }

        try:
            logger.info(f"Fetching forecast weather from {self.config.base_url}")
            response = requests.get(self.config.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._process_response(data)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather forecast: {e}")
            raise

    def _process_response(self, data: dict) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly:
            raise ValueError("No hourly data in response")

        # Required columns for core logic
        cols = {
            "ds": pd.to_datetime(hourly["time"], utc=True),
            "temperature_2m": hourly.get("temperature_2m"),
            "cloud_cover": hourly.get("cloud_cover"),
            "shortwave_radiation": hourly.get("shortwave_radiation"),
            "snow_depth": hourly.get("snow_depth"),
            "snowfall": hourly.get("snowfall"),
        }

        df = pd.DataFrame(cols)

        # Fill missing snow data with 0 (essential for constraints)
        for snow_col in ["snow_depth", "snowfall"]:
            if snow_col in df.columns:
                df[snow_col] = df[snow_col].fillna(0)

        # Open-Meteo returns nulls sometimes, fill or drop?
        # Linear interpolation is usually safe for weather gaps
        df = df.interpolate(method="linear").bfill().ffill()

        return df

    def resample_weather(self, df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
        """
        Upsample hourly weather data to matching interval (e.g. 15 min).
        Uses cubic interpolation for smooth curves.
        """
        if df.empty:
            return df

        df = df.set_index("ds").sort_index()

        # Check if we need resampling
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        current_freq = pd.infer_freq(df.index)
        target_freq = f"{interval_minutes}min"

        if current_freq == target_freq:
            return df.reset_index()

        # Resample and interpolate
        # Cubic is good for temperature/radiation curves
        # Note: Snow depth/fall might be better with linear or ffill,
        # but cubic handles radiation/temp beautifully.
        df_resampled = df.resample(target_freq).interpolate(method="cubic")

        # Reset index to get 'ds' column back
        return df_resampled.reset_index()
