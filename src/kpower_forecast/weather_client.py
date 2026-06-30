"""Open-Meteo-compatible weather client with model fallback and caching."""

import datetime
import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

INVALID_VARIABLE_PATTERN = re.compile(r"invalid String value\s+([a-zA-Z0-9_]+)")
DEFAULT_WEATHER_CACHE_DIR = Path(".kpower_weather_cache")
MINUTELY_15 = "minutely_15"
HOURLY = "hourly"
FORECAST_HOURS_PER_DAY = 24
WEATHER_REQUEST_INTERVAL_MINUTES = 15
FORECAST_INTERVALS_PER_HOUR = 60 // WEATHER_REQUEST_INTERVAL_MINUTES

PHYSICAL_LOWER_BOUNDED_COLUMNS: tuple[str, ...] = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "snow_depth",
    "snowfall",
    "precipitation",
    "rain",
    "showers",
    "snowfall_convective_water_equivalent",
    "snowfall_water_equivalent",
    "snowfall_height",
)


class WeatherConfig(BaseModel):
    """Configuration for Open-Meteo-compatible weather requests.

    Args:
        base_url: Forecast API URL.
        archive_url: Historical archive API URL.
        forecast_model: Optional primary forecast model identifier. When unset,
            Open-Meteo chooses the best matching model.
        long_horizon_model: Optional model used to fill a short primary forecast.
        cache_enabled: Whether weather API responses should be cached on disk.
        cache_dir: Optional directory for cached weather responses.
        forecast_cache_ttl_hours: Forecast cache freshness window in hours.
        historical_cache_ttl_hours: Historical cache freshness window in hours.
        required_hourly_variables: Variables required by the forecasting core.
        optional_hourly_variables: Extra variables requested when the provider
            supports them.
    """

    base_url: str = "https://api.open-meteo.com/v1/forecast"
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_model: Optional[str] = None
    long_horizon_model: Optional[str] = "ecmwf_ifs"
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    forecast_cache_ttl_hours: float = Field(default=1.0, gt=0)
    historical_cache_ttl_hours: float = Field(default=720.0, gt=0)
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
            "snowfall_water_equivalent",
            "snowfall_height",
        ]
    )

    @property
    def hourly_variables(self) -> list[str]:
        """Return de-duplicated weather variables for API requests.

        Returns:
            List of weather variable names.
        """
        return list(
            dict.fromkeys(
                self.required_hourly_variables + self.optional_hourly_variables
            )
        )

    @field_validator("forecast_model", "long_horizon_model")
    @classmethod
    def normalize_optional_model(cls, value: Optional[str]) -> Optional[str]:
        """Normalize empty model strings to ``None``.

        Args:
            value: Optional weather model identifier.

        Returns:
            Normalized model identifier, or None.
        """
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class WeatherClient:
    """Client for Open-Meteo-compatible weather APIs."""

    def __init__(self, lat: float, lon: float, config: Optional[WeatherConfig] = None):
        self.lat = lat
        self.lon = lon
        self.config = config or WeatherConfig()

    def effective_forecast_model_id(self) -> str:
        """Return the stable model identifier for forecast metadata.

        Returns:
            Model identifier including long-horizon fallback when configured.
        """
        primary_model = self.config.forecast_model or "best_match"
        long_horizon_model = self.config.long_horizon_model
        if long_horizon_model and long_horizon_model != self.config.forecast_model:
            return f"{primary_model}+{long_horizon_model}"
        return primary_model

    def fetch_historical(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for training.
        """
        weather_variables = list(self.config.hourly_variables)
        request_field = MINUTELY_15

        while True:
            params: dict[str, str | float | list[str]] = {
                "latitude": self.lat,
                "longitude": self.lon,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                request_field: weather_variables,
                "timezone": "UTC",
            }

            try:
                logger.info(
                    f"Fetching historical weather from {self.config.archive_url}"
                )
                data = self._request_json(
                    endpoint="historical",
                    url=self.config.archive_url,
                    params=params,
                    ttl_hours=self.config.historical_cache_ttl_hours,
                )
                return self._process_response(data)
            except requests.HTTPError as error:
                invalid_variable = self._extract_invalid_variable(error)
                if invalid_variable and invalid_variable in weather_variables:
                    logger.warning(
                        "Archive API rejected variable '%s'. Retrying without it.",
                        invalid_variable,
                    )
                    weather_variables = [
                        variable
                        for variable in weather_variables
                        if variable != invalid_variable
                    ]
                    if not weather_variables:
                        logger.error(
                            "Archive API rejected all weather variables "
                            "for historical fetch."
                        )
                        raise
                    continue
                if self._should_retry_hourly(error, request_field):
                    request_field = HOURLY
                    logger.warning(
                        "Archive API rejected 15-minute weather data. "
                        "Retrying with hourly data."
                    )
                    continue

                logger.error(f"Failed to fetch historical weather: {error}")
                raise
            except requests.RequestException as error:
                logger.error(f"Failed to fetch historical weather: {error}")
                raise

    def _extract_invalid_variable(self, error: requests.HTTPError) -> Optional[str]:
        """Extract unsupported variable name from Open-Meteo error payload."""
        response = error.response
        if response is None:
            return None

        try:
            payload = response.json()
        except ValueError:
            return None

        reason = payload.get("reason") if isinstance(payload, dict) else None
        if not isinstance(reason, str):
            return None

        match = INVALID_VARIABLE_PATTERN.search(reason)
        return match.group(1) if match else None

    def fetch_forecast(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch weather forecast for prediction.
        """
        weather_variables = list(self.config.hourly_variables)
        request_field = MINUTELY_15

        while True:
            params = self._build_forecast_params(
                weather_variables=weather_variables,
                days=days,
                model=self.config.forecast_model,
                request_field=request_field,
            )

            try:
                logger.info(f"Fetching forecast weather from {self.config.base_url}")
                primary_data = self._request_json(
                    endpoint="forecast",
                    url=self.config.base_url,
                    params=params,
                    ttl_hours=self.config.forecast_cache_ttl_hours,
                )
                primary = self._process_response(primary_data, interpolate=False)
                merged = self._maybe_fill_long_horizon(
                    primary=primary,
                    weather_variables=weather_variables,
                    days=days,
                    request_field=request_field,
                )
                return self._finalize_weather_frame(merged)
            except requests.HTTPError as error:
                invalid_variable = self._extract_invalid_variable(error)
                if invalid_variable and invalid_variable in weather_variables:
                    logger.warning(
                        "Forecast API rejected variable '%s'. Retrying without it.",
                        invalid_variable,
                    )
                    weather_variables = [
                        variable
                        for variable in weather_variables
                        if variable != invalid_variable
                    ]
                    if not weather_variables:
                        logger.error(
                            "Forecast API rejected all weather variables "
                            "for forecast fetch."
                        )
                        raise
                    continue
                if self._should_retry_hourly(error, request_field):
                    request_field = HOURLY
                    logger.warning(
                        "Forecast API rejected 15-minute weather data. "
                        "Retrying with hourly data."
                    )
                    continue

                logger.error(f"Failed to fetch weather forecast: {error}")
                raise
            except requests.RequestException as error:
                logger.error(f"Failed to fetch weather forecast: {error}")
                raise

    def _should_retry_hourly(
        self, error: requests.HTTPError, request_field: str
    ) -> bool:
        """Return whether a 15-minute request should fall back to hourly data.

        Args:
            error: HTTP error raised by the weather API.
            request_field: Current weather variable request field.

        Returns:
            True when the request can safely retry with ``hourly``.
        """
        response = error.response
        return (
            request_field == MINUTELY_15
            and response is not None
            and response.status_code == 400
        )

    def _build_forecast_params(
        self,
        weather_variables: list[str],
        days: int,
        model: Optional[str],
        request_field: str,
    ) -> dict[str, str | float | int | list[str]]:
        """Build Open-Meteo forecast query parameters.

        Args:
            weather_variables: Weather variable names to request.
            days: Requested forecast horizon in days.
            model: Optional Open-Meteo model identifier.
            request_field: Open-Meteo resolution field.

        Returns:
            Query parameter dictionary.
        """
        params: dict[str, str | float | int | list[str]] = {
            "latitude": self.lat,
            "longitude": self.lon,
            request_field: weather_variables,
            "forecast_days": days,
            "timezone": "UTC",
        }
        if model:
            params["models"] = model
        return params

    def _maybe_fill_long_horizon(
        self,
        primary: pd.DataFrame,
        weather_variables: list[str],
        days: int,
        request_field: str,
    ) -> pd.DataFrame:
        """Fill short primary forecasts with the configured long-horizon model.

        Args:
            primary: Primary forecast dataframe.
            weather_variables: Weather variable names used by the primary request.
            days: Requested forecast horizon in days.
            request_field: Open-Meteo resolution field used by the primary request.

        Returns:
            Primary dataframe, optionally filled by the long-horizon dataframe.
        """
        expected_rows = self._expected_forecast_rows(days, request_field)
        has_full_horizon = self._forecast_row_count(primary) >= expected_rows
        has_weather_data = self._has_required_weather_data(primary)
        if has_full_horizon and has_weather_data:
            return primary

        long_horizon_model = self.config.long_horizon_model
        if not long_horizon_model or long_horizon_model == self.config.forecast_model:
            self._warn_partial_forecast(primary, days, request_field)
            return primary

        if not has_full_horizon:
            logger.warning(
                "Primary weather forecast returned %s rows for requested "
                "%s-day horizon. Fetching long-horizon model '%s'.",
                self._forecast_row_count(primary),
                days,
                long_horizon_model,
            )
        else:
            logger.warning(
                "Primary weather forecast returned no usable required weather "
                "data. Fetching long-horizon model '%s'.",
                long_horizon_model,
            )
        params = self._build_forecast_params(
            weather_variables=weather_variables,
            days=days,
            model=long_horizon_model,
            request_field=request_field,
        )
        long_data = self._request_json(
            endpoint="forecast",
            url=self.config.base_url,
            params=params,
            ttl_hours=self.config.forecast_cache_ttl_hours,
        )
        long_horizon = self._process_response(long_data, interpolate=False)
        merged = self._merge_weather_frames(primary=primary, fallback=long_horizon)
        self._warn_partial_forecast(merged, days, request_field)
        return merged

    def _has_required_weather_data(self, df: pd.DataFrame) -> bool:
        """Return whether a weather frame has usable required weather values.

        Args:
            df: Weather dataframe.

        Returns:
            True when at least one required weather column contains data.
        """
        required_columns = [
            column for column in self.config.required_hourly_variables if column in df
        ]
        if not required_columns:
            return False
        required_values = df[required_columns].apply(pd.to_numeric, errors="coerce")
        return bool(required_values.notna().any().any())

    def _expected_forecast_rows(self, days: int, request_field: str) -> int:
        """Return the expected forecast row count.

        Args:
            days: Forecast horizon in days.
            request_field: Open-Meteo resolution field.

        Returns:
            Number of rows expected for the requested horizon.
        """
        intervals_per_hour = (
            FORECAST_INTERVALS_PER_HOUR if request_field == MINUTELY_15 else 1
        )
        return max(days, 0) * FORECAST_HOURS_PER_DAY * intervals_per_hour

    def _forecast_row_count(self, df: pd.DataFrame) -> int:
        """Return the number of unique forecast timestamps in a dataframe.

        Args:
            df: Forecast dataframe.

        Returns:
            Unique timestamp count, or row count when ``ds`` is unavailable.
        """
        if "ds" not in df.columns:
            return len(df)
        return int(pd.to_datetime(df["ds"], utc=True).nunique())

    def _warn_partial_forecast(
        self, df: pd.DataFrame, days: int, request_field: str = MINUTELY_15
    ) -> None:
        """Log a warning when returned forecast rows are below the request.

        Args:
            df: Forecast dataframe.
            days: Requested forecast horizon in days.
            request_field: Open-Meteo resolution field.

        Returns:
            None.
        """
        expected_rows = self._expected_forecast_rows(days, request_field)
        row_count = self._forecast_row_count(df)
        if row_count >= expected_rows:
            return
        logger.warning(
            "Weather forecast returned %s rows, below requested %s rows "
            "for a %s-day horizon. Returning partial weather data.",
            row_count,
            expected_rows,
            days,
        )

    def _merge_weather_frames(
        self, primary: pd.DataFrame, fallback: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge weather frames, preferring primary non-null values.

        Args:
            primary: Primary weather dataframe.
            fallback: Fallback weather dataframe.

        Returns:
            Merged weather dataframe sorted by timestamp.

        Raises:
            ValueError: If either dataframe is missing ``ds``.
        """
        if "ds" not in primary.columns or "ds" not in fallback.columns:
            raise ValueError("weather dataframes must contain a 'ds' column")

        primary_indexed = primary.copy()
        fallback_indexed = fallback.copy()
        primary_indexed["ds"] = pd.to_datetime(primary_indexed["ds"], utc=True)
        fallback_indexed["ds"] = pd.to_datetime(fallback_indexed["ds"], utc=True)
        primary_indexed = primary_indexed.drop_duplicates(subset=["ds"], keep="last")
        fallback_indexed = fallback_indexed.drop_duplicates(subset=["ds"], keep="last")

        merged = primary_indexed.set_index("ds").combine_first(
            fallback_indexed.set_index("ds")
        )
        return merged.sort_index().reset_index()

    def _request_json(
        self,
        endpoint: str,
        url: str,
        params: dict[str, str | float | int | list[str]],
        ttl_hours: float,
    ) -> dict[str, object]:
        """Fetch JSON from cache or the weather API.

        Args:
            endpoint: Logical endpoint name for cache-key separation.
            url: Weather API URL.
            params: Query parameters.
            ttl_hours: Cache time-to-live in hours.

        Returns:
            Raw JSON response payload.

        Raises:
            requests.RequestException: If the HTTP request fails.
            ValueError: If the API response is not a JSON object.
        """
        cached = self._load_cached_response(endpoint, url, params)
        if cached is not None:
            return cached

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Weather API response must be a JSON object")
        payload = cast(dict[str, object], data)
        self._store_cached_response(endpoint, url, params, payload, ttl_hours)
        return payload

    def _cache_path(
        self,
        endpoint: str,
        url: str,
        params: dict[str, str | float | int | list[str]],
    ) -> Path:
        """Build the cache path for a weather request.

        Args:
            endpoint: Logical endpoint name for cache-key separation.
            url: Weather API URL.
            params: Query parameters.

        Returns:
            Cache file path.
        """
        cache_dir = self.config.cache_dir or DEFAULT_WEATHER_CACHE_DIR
        normalized = {
            "endpoint": endpoint,
            "url": url,
            "params": params,
        }
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return cache_dir / f"{endpoint}_{cache_key}.json"

    def _load_cached_response(
        self,
        endpoint: str,
        url: str,
        params: dict[str, str | float | int | list[str]],
    ) -> Optional[dict[str, object]]:
        """Load a fresh cached weather response if available.

        Args:
            endpoint: Logical endpoint name for cache-key separation.
            url: Weather API URL.
            params: Query parameters.

        Returns:
            Cached raw response payload, or None.
        """
        if not self.config.cache_enabled:
            return None

        cache_path = self._cache_path(endpoint, url, params)
        if not cache_path.exists():
            return None

        try:
            with cache_path.open(encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                return None
            created_at_raw = payload.get("created_at")
            ttl_seconds_raw = payload.get("ttl_seconds")
            data = payload.get("data")
            if (
                not isinstance(created_at_raw, str)
                or not isinstance(ttl_seconds_raw, int | float)
                or not isinstance(data, dict)
            ):
                return None

            created_at = datetime.datetime.fromisoformat(created_at_raw)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=datetime.timezone.utc)
            expires_at = created_at + datetime.timedelta(seconds=float(ttl_seconds_raw))
            if datetime.datetime.now(datetime.timezone.utc) >= expires_at:
                return None
            logger.debug(
                "Using cached %s weather response from %s", endpoint, cache_path
            )
            return cast(dict[str, object], data)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            logger.warning(
                "Ignoring invalid weather cache entry %s: %s", cache_path, error
            )
            return None

    def _store_cached_response(
        self,
        endpoint: str,
        url: str,
        params: dict[str, str | float | int | list[str]],
        data: dict[str, object],
        ttl_hours: float,
    ) -> None:
        """Store a raw weather response in the on-disk cache.

        Args:
            endpoint: Logical endpoint name for cache-key separation.
            url: Weather API URL.
            params: Query parameters.
            data: Raw response payload.
            ttl_hours: Cache time-to-live in hours.

        Returns:
            None.
        """
        if not self.config.cache_enabled:
            return

        cache_path = self._cache_path(endpoint, url, params)
        cache_payload = {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ttl_seconds": ttl_hours * 3600.0,
            "data": data,
        }
        temp_path = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(cache_payload, file, sort_keys=True)
            temp_path.replace(cache_path)
        except OSError as error:
            logger.warning(
                "Failed to write weather cache entry %s: %s", cache_path, error
            )

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

    def _weather_payload(self, data: dict) -> dict[str, object]:
        """Return the highest-resolution weather payload from the response.

        Args:
            data: Raw JSON payload from the weather API.

        Returns:
            Weather payload dictionary.

        Raises:
            ValueError: If no supported weather payload is available.
        """
        for key in (MINUTELY_15, HOURLY):
            payload = data.get(key)
            if isinstance(payload, dict):
                return cast(dict[str, object], payload)
        raise ValueError("No weather data in response")

    def _parse_weather_times(self, data: dict) -> pd.Series:
        """Parse weather timestamps and normalize them to UTC.

        Args:
            data: Raw JSON payload from the weather API.

        Returns:
            Series of UTC-aware timestamps.

        Raises:
            ValueError: If the response does not contain weather timestamps.
        """
        weather_payload = self._weather_payload(data)
        if "time" not in weather_payload:
            raise ValueError("No weather time data in response")

        raw_times = weather_payload["time"]
        if not isinstance(raw_times, list):
            raise ValueError("Weather time data must be a list")

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

    def _process_response(self, data: dict, interpolate: bool = True) -> pd.DataFrame:
        """Convert an Open-Meteo response into a weather dataframe.

        Args:
            data: Raw JSON payload from the weather API.
            interpolate: Whether to fill gaps after parsing.

        Returns:
            DataFrame with UTC timestamps and numeric weather columns.

        Raises:
            ValueError: If the response does not contain weather data.
        """
        weather_payload = self._weather_payload(data)

        timestamps = self._parse_weather_times(data)
        cols: dict[str, object] = {"ds": timestamps}
        for variable in self.config.required_hourly_variables:
            values = weather_payload.get(variable)
            cols[variable] = values if values is not None else [None] * len(timestamps)

        for variable in self.config.optional_hourly_variables:
            if variable in weather_payload:
                cols[variable] = weather_payload[variable]

        df = pd.DataFrame(cols)

        # Coerce all numeric columns to float64 — Open-Meteo may return
        # all-null lists (e.g. snow_depth on reduced datasets) which pandas
        # stores as object dtype; interpolate() refuses object columns.
        numeric_cols = [c for c in df.columns if c != "ds"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        if not interpolate:
            return self._clip_physical_bounds(df)

        return self._finalize_weather_frame(df)

    def _finalize_weather_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill small weather gaps and apply physical bounds.

        Args:
            df: Parsed weather dataframe.

        Returns:
            Weather dataframe with interpolated numeric columns.
        """
        output = df.copy()
        output["ds"] = pd.to_datetime(output["ds"], utc=True)
        output = (
            output.drop_duplicates(subset=["ds"], keep="last")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        numeric_cols = [c for c in output.columns if c != "ds"]
        output[numeric_cols] = output[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        # Fill missing snow data with 0 (essential for constraints)
        for snow_col in ["snow_depth", "snowfall"]:
            if snow_col in output.columns:
                output[snow_col] = output[snow_col].fillna(0.0)

        # Open-Meteo returns nulls sometimes, fill or drop?
        # Linear interpolation is usually safe for weather gaps
        output = output.interpolate(method="linear").bfill().ffill()

        return self._clip_physical_bounds(output)

    def _clip_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip weather variables that cannot be physically negative.

        Args:
            df: Weather dataframe with optional Open-Meteo columns.

        Returns:
            Copy of ``df`` with lower-bounded physical variables clipped to zero.
        """
        bounded_columns = [
            column for column in PHYSICAL_LOWER_BOUNDED_COLUMNS if column in df.columns
        ]
        if not bounded_columns:
            return df

        clipped = df.copy()
        clipped[bounded_columns] = clipped[bounded_columns].clip(lower=0.0)
        return clipped

    def resample_weather(self, df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
        """
        Resample weather data to the configured forecast interval.
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
            return self._clip_physical_bounds(df.reset_index())

        # Resample and interpolate
        # Cubic is good for temperature/radiation curves
        # Note: Snow depth/fall might be better with linear or ffill,
        # but cubic handles radiation/temp beautifully.
        df_resampled = df.resample(target_freq).interpolate(method="cubic")

        # Reset index to get 'ds' column back
        return self._clip_physical_bounds(df_resampled.reset_index())
