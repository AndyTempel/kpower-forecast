import datetime
import logging
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

    def _get_response_timezone(self, data: dict) -> datetime.tzinfo:
        """Resolve the timezone metadata returned by Open-Meteo-compatible APIs.

        Args:
            data: Raw JSON payload from the weather API.

        Returns:
            Timezone used by naive hourly timestamp strings.
        """
        timezone_name = data.get("timezone")
        if isinstance(timezone_name, str):
            if timezone_name.upper() in {"GMT", "UTC"}:
                return datetime.timezone.utc
            try:
                return ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                pass

        offset_seconds = int(data.get("utc_offset_seconds") or 0)
        return datetime.timezone(datetime.timedelta(seconds=offset_seconds))

    def _parse_hourly_times(self, data: dict) -> pd.Series:
        """Parse hourly timestamps and normalize them to UTC.

        Args:
            data: Raw JSON payload from the weather API.

        Returns:
            Series of UTC-aware timestamps.

        Raises:
            ValueError: If the response does not contain hourly timestamps.
        """
        hourly = data.get("hourly", {})
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise ValueError("No hourly time data in response")

        raw_times = hourly["time"]
        if not isinstance(raw_times, list):
            raise ValueError("Hourly time data must be a list")

        time_series = pd.to_datetime(
            pd.Series(raw_times, dtype="object"), errors="raise"
        )
        if time_series.dt.tz is not None:
            return time_series.dt.tz_convert("UTC")

        response_tz = self._get_response_timezone(data)
        if response_tz != datetime.timezone.utc:
            logger.warning(
                "Weather response timestamps are %s; converting them to UTC.",
                data.get("timezone", response_tz),
            )
        return time_series.dt.tz_localize(response_tz).dt.tz_convert("UTC")

    def _process_response(self, data: dict) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly:
            raise ValueError("No hourly data in response")

        # Required columns for core logic
        cols = {
            "ds": self._parse_hourly_times(data),
            "temperature_2m": hourly.get("temperature_2m"),
            "cloud_cover": hourly.get("cloud_cover"),
            "shortwave_radiation": hourly.get("shortwave_radiation"),
            "snow_depth": hourly.get("snow_depth"),
            "snowfall": hourly.get("snowfall"),
        }

        df = pd.DataFrame(cols)

        # Coerce all numeric columns to float64 — Open-Meteo may return
        # all-null lists (e.g. snow_depth on reduced datasets) which pandas
        # stores as object dtype; interpolate() refuses object columns.
        numeric_cols = [c for c in df.columns if c != "ds"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Fill missing snow data with 0 (essential for constraints)
        for snow_col in ["snow_depth", "snowfall"]:
            if snow_col in df.columns:
                df[snow_col] = df[snow_col].fillna(0.0)

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
