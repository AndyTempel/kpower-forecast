import datetime
import logging
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WeatherConfig(BaseModel):
    """Configuration for Open-Meteo-compatible weather requests.

    Args:
        base_url: Forecast API URL.
        archive_url: Historical archive API URL.
        forecast_model: Optional forecast model identifier passed to Open-Meteo.
        required_hourly_variables: Variables required by the forecasting core.
        optional_hourly_variables: Extra variables requested when the provider
            supports them.
    """

    base_url: str = "https://api.open-meteo.com/v1/forecast"
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_model: Optional[str] = "ecmwf_ifs"
    required_hourly_variables: list[str] = Field(
        default_factory=lambda: [
            "temperature_2m",
            "cloud_cover",
            "shortwave_radiation",
            "snow_depth",
            "snowfall",
        ]
    )
    optional_hourly_variables: list[str] = Field(
        default_factory=lambda: [
            "precipitation",
            "direct_radiation",
            "diffuse_radiation",
            "wind_u_component_10m",
            "wind_v_component_10m",
            "wind_gusts_10m",
            "rain",
            "showers",
            "weather_code",
            "snowfall_convective_water_equivalent",
            "snowfall_water_equivalent",
            "snowfall_height",
        ]
    )

    @property
    def hourly_variables(self) -> list[str]:
        """Return de-duplicated hourly variables for API requests.

        Returns:
            List of hourly variable names.
        """
        return list(
            dict.fromkeys(
                self.required_hourly_variables + self.optional_hourly_variables
            )
        )


class WeatherClient:
    """Client for Open-Meteo-compatible weather APIs."""

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
            "hourly": self.config.hourly_variables,
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
            "hourly": self.config.hourly_variables,
            "forecast_days": days,
            "timezone": "UTC",
        }
        if self.config.forecast_model:
            params["models"] = self.config.forecast_model

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
        """Convert an Open-Meteo response into a weather dataframe.

        Args:
            data: Raw JSON payload from the weather API.

        Returns:
            DataFrame with UTC timestamps and numeric weather columns.

        Raises:
            ValueError: If the response does not contain hourly data.
        """
        hourly = data.get("hourly", {})
        if not hourly:
            raise ValueError("No hourly data in response")

        timestamps = self._parse_hourly_times(data)
        cols: dict[str, object] = {"ds": timestamps}
        for variable in self.config.required_hourly_variables:
            values = hourly.get(variable)
            cols[variable] = values if values is not None else [None] * len(timestamps)

        for variable in self.config.optional_hourly_variables:
            if variable in hourly:
                cols[variable] = hourly[variable]

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
