import logging
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import ModelStorage
from .utils import calculate_solar_elevation, get_clear_sky_ghi
from .weather_client import WeatherClient, WeatherConfig

logger = logging.getLogger(__name__)


class MeasurementUnit(str, Enum):
    KWH = "kWh"
    WH = "Wh"
    KW = "kW"
    W = "W"


class DataCategory(str, Enum):
    INSTANT_ENERGY = "instant_energy"  # Energy per interval (kWh/Wh)
    CUMULATIVE_ENERGY = "cumulative_energy"  # Meter reading (kWh/Wh)
    POWER = "power"  # Instantaneous power (kW/W)


class PredictionInterval(BaseModel):
    timestamp: pd.Timestamp
    expected_kwh: float
    lower_bound_kwh: float  # P10
    upper_bound_kwh: float  # P90

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KPowerConfig(BaseModel):
    """Runtime configuration for a forecasting model."""

    model_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    storage_path: str = "./data"
    interval_minutes: int = Field(15)
    forecast_type: Literal["solar", "consumption"] = "solar"
    data_category: DataCategory = DataCategory.INSTANT_ENERGY
    unit: MeasurementUnit = MeasurementUnit.KWH
    heat_pump_mode: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    efficiency_factor: Optional[float] = None  # Learned ratio: kW / (W/m2)
    efficiency_profile: Optional[dict[int, float]] = None
    max_efficiency_factor: Optional[float] = None
    efficiency_cap_headroom: float = Field(default=1.15, gt=0)
    adaptive_weather_correction: bool = True
    weather_correction: Optional[dict[str, Any]] = None
    min_weather_correction_samples: int = Field(default=8, gt=0)
    min_weather_correction_multiplier: float = Field(default=0.65, gt=0)
    max_weather_correction_multiplier: float = Field(default=1.35, gt=0)
    inverter_ac_limit_kw: Optional[float] = Field(default=None, gt=0)
    grid_export_limit_kw: Optional[float] = Field(default=None, gt=0)

    @field_validator("interval_minutes")
    @classmethod
    def check_interval(cls, v: int) -> int:
        if v not in (15, 60):
            raise ValueError("interval_minutes must be 15 or 60")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KPowerForecast:
    """Train and serve solar production or consumption forecasts."""

    def __init__(
        self,
        model_id: str,
        latitude: float,
        longitude: float,
        storage_path: str = "./data",
        interval_minutes: int = 15,
        forecast_type: Literal["solar", "consumption"] = "solar",
        data_category: DataCategory = DataCategory.INSTANT_ENERGY,
        unit: MeasurementUnit = MeasurementUnit.KWH,
        heat_pump_mode: bool = False,
        adaptive_weather_correction: bool = True,
        inverter_ac_limit_kw: Optional[float] = None,
        grid_export_limit_kw: Optional[float] = None,
        weather_config: Optional[WeatherConfig] = None,
    ):
        self.config = KPowerConfig(
            model_id=model_id,
            latitude=latitude,
            longitude=longitude,
            storage_path=storage_path,
            interval_minutes=interval_minutes,
            forecast_type=forecast_type,
            data_category=data_category,
            unit=unit,
            heat_pump_mode=heat_pump_mode,
            adaptive_weather_correction=adaptive_weather_correction,
            inverter_ac_limit_kw=inverter_ac_limit_kw,
            grid_export_limit_kw=grid_export_limit_kw,
        )

        self.weather_client = WeatherClient(
            lat=self.config.latitude,
            lon=self.config.longitude,
            config=self._weather_config_with_default_cache(weather_config),
        )
        self.storage = ModelStorage(storage_path=self.config.storage_path)
        self._model: Optional[Prophet] = None

    def _weather_config_with_default_cache(
        self, weather_config: Optional[WeatherConfig]
    ) -> WeatherConfig:
        """Return weather config with a storage-scoped default cache directory.

        Args:
            weather_config: Optional caller-provided weather configuration.

        Returns:
            Weather configuration for this forecast instance.
        """
        default_cache_dir = Path(self.config.storage_path) / "weather_cache"
        if weather_config is None:
            return WeatherConfig(cache_dir=default_cache_dir)
        if weather_config.cache_enabled and weather_config.cache_dir is None:
            return weather_config.model_copy(update={"cache_dir": default_cache_dir})
        return weather_config

    def _apply_model_metadata(self, metadata: dict[str, Any]) -> None:
        """Apply persisted model metadata to the runtime configuration.

        Args:
            metadata: Metadata dictionary loaded from model storage.

        Returns:
            None.
        """
        self.config.efficiency_factor = metadata.get("efficiency_factor")
        self.config.efficiency_profile = self._normalize_efficiency_profile(
            metadata.get("efficiency_profile")
        )
        self.config.weather_correction = metadata.get("weather_correction")

    def _build_model_metadata(self) -> dict[str, Any]:
        """Build JSON-serializable metadata for model persistence.

        Returns:
            Dictionary containing learned solar calibration metadata.
        """
        return {
            "efficiency_factor": self.config.efficiency_factor,
            "efficiency_profile": self.config.efficiency_profile,
            "weather_correction": self.config.weather_correction,
            "forecast_model": self._forecast_model_id(),
        }

    def _forecast_model_id(self) -> str:
        """Return a stable weather model identifier for metadata and storage.

        Returns:
            Weather model identifier, including long-horizon fallback when configured.
        """
        return self.weather_client.effective_forecast_model_id()

    def _normalize_efficiency_profile(self, profile: Any) -> Optional[dict[int, float]]:
        """Normalize persisted efficiency profile keys and values.

        Args:
            profile: Raw profile object loaded from storage.

        Returns:
            Efficiency profile keyed by minute of day, or None when invalid.

        Raises:
            ValueError: If the stored profile contains invalid keys or values.
        """
        if profile is None:
            return None
        if not isinstance(profile, dict):
            raise ValueError("efficiency_profile metadata must be a dictionary")

        normalized: dict[int, float] = {}
        for raw_key, raw_value in profile.items():
            minute_of_day = int(raw_key)
            factor = float(raw_value)
            if minute_of_day < 0 or minute_of_day >= 1440:
                raise ValueError("efficiency_profile keys must be minutes of day")
            if factor <= 0:
                raise ValueError("efficiency_profile values must be positive")
            normalized[minute_of_day] = factor

        return normalized or None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-informed features and rolling windows.
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"], utc=True)

        # 1. Physics: Clear Sky GHI
        logger.info("Calculating physics-informed Clear Sky GHI...")
        temp_df = df.set_index("ds")
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        df["clear_sky_ghi"] = get_clear_sky_ghi(
            self.config.latitude, self.config.longitude, temp_df.index
        ).values

        # 2. Rolling Cloud Cover (3-hour window)
        window_size = 180 // self.config.interval_minutes
        logger.info(f"Adding rolling cloud cover (window={window_size})...")
        df["rolling_cloud_cover"] = (
            df["cloud_cover"].rolling(window=window_size, min_periods=1).mean()
        )

        return df

    def _prepare_training_data(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for training or tuning by merging with weather data.
        """
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

        # Standard weather columns + snow variables
        weather_cols = [
            "temperature_2m",
            "cloud_cover",
            "shortwave_radiation",
            "snow_depth",
            "snowfall",
        ]
        # Only interpolate/fill if columns exist
        available_weather = [c for c in weather_cols if c in df.columns]
        df[available_weather] = (
            df[available_weather].interpolate(method="linear").bfill().ffill()
        )

        # Drop only if essential columns are missing
        essential = ["temperature_2m", "cloud_cover", "shortwave_radiation"]
        df = df.dropna(subset=essential)

        return self._prepare_features(df)

    def calibrate_efficiency(self, df: pd.DataFrame) -> Optional[float]:
        """
        Learns the system's effective efficiency factor from historical data.
        Factor = kW / Irradiance (W/m2).
        Uses 95th percentile under clear sky conditions to find effective peak.
        This factor represents kW capacity per W/m2 and is time-invariant.
        """
        if self.config.forecast_type != "solar":
            return None

        # Filter for clear sky (<25% cloud cover) and non-zero irradiance
        # NEW: Filter out snowy days (snow_depth > 0.01m)
        mask = (df["cloud_cover"] < 25) & (df["shortwave_radiation"] > 50)
        if "snow_depth" in df.columns:
            mask &= df["snow_depth"] <= 0.01

        clear_sky_df = df[mask].copy()

        if clear_sky_df.empty or len(clear_sky_df) < 10:
            logger.warning("Insufficient clear sky data for efficiency calibration.")
            return None

        # Convert Energy (kWh per interval) to Power (kW)
        interval_hours = self.config.interval_minutes / 60.0
        power_kw = clear_sky_df["y"] / interval_hours

        # Calculate ratio: kW per W/m2
        clear_sky_df["eff"] = power_kw / clear_sky_df["shortwave_radiation"]

        if clear_sky_df.empty:
            return None

        # 95th percentile finds peak performance while ignoring sensor noise
        factor = float(clear_sky_df["eff"].quantile(0.95))

        # Optional safety cap. Defaults to None because PV fleet sizes can range
        # from home systems to multi-megawatt industrial installations.
        if (
            self.config.max_efficiency_factor is not None
            and factor > self.config.max_efficiency_factor
        ):
            logger.warning(
                f"Calibrated factor {factor:.6f} exceeds safety limit "
                f"{self.config.max_efficiency_factor}. Clamping."
            )
            factor = self.config.max_efficiency_factor

        logger.info(f"Calibrated efficiency factor: {factor:.6f} kW per W/m2")
        return factor

    def calibrate_efficiency_profile(
        self, df: pd.DataFrame
    ) -> Optional[dict[int, float]]:
        """Learns a time-of-day solar efficiency envelope from history.

        The global efficiency factor is a useful fallback, but asymmetric arrays
        can produce much more power per horizontal irradiance in the morning than
        in the afternoon. This profile learns that envelope directly from clear
        historical production without requiring panel azimuth or tilt.

        Args:
            df: Prepared training dataframe with production and weather columns.

        Returns:
            Dictionary keyed by UTC minute of day with kW per W/m2 factors, or
            None when calibration data is insufficient.
        """
        if self.config.forecast_type != "solar":
            return None

        required_columns = {"ds", "y", "cloud_cover", "shortwave_radiation"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "df is missing required columns for efficiency profile: "
                f"{sorted(missing_columns)}"
            )

        mask = (df["cloud_cover"] < 25) & (df["shortwave_radiation"] > 50)
        if "snow_depth" in df.columns:
            mask &= df["snow_depth"] <= 0.01

        clear_sky_df = df[mask].copy()
        if clear_sky_df.empty or len(clear_sky_df) < 10:
            logger.warning("Insufficient clear sky data for efficiency profile.")
            return None

        clear_sky_df["ds"] = pd.to_datetime(clear_sky_df["ds"], utc=True)
        interval_hours = self.config.interval_minutes / 60.0
        power_kw = clear_sky_df["y"] / interval_hours
        clear_sky_df["eff"] = power_kw / clear_sky_df["shortwave_radiation"]
        clear_sky_df = clear_sky_df[clear_sky_df["eff"] > 0]

        if clear_sky_df.empty:
            return None

        global_factor = self.config.efficiency_factor
        if global_factor is None:
            global_factor = float(clear_sky_df["eff"].quantile(0.95))

        clear_sky_df["minute_of_day"] = (
            clear_sky_df["ds"].dt.hour * 60 + clear_sky_df["ds"].dt.minute
        )
        grouped = clear_sky_df.groupby("minute_of_day")["eff"].quantile(0.95)
        slots = pd.Index(range(0, 1440, self.config.interval_minutes))
        profile = grouped.reindex(slots).interpolate(limit_direction="both")
        profile = profile.fillna(global_factor)
        profile = profile.clip(lower=global_factor)
        profile = profile.rolling(window=3, center=True, min_periods=1).max()

        if self.config.max_efficiency_factor is not None:
            profile = profile.clip(upper=self.config.max_efficiency_factor)

        learned_profile: dict[int, float] = {}
        for slot, value in profile.items():
            learned_profile[int(cast(int, slot))] = float(value)
        logger.info(
            "Calibrated efficiency profile with %s slots.", len(learned_profile)
        )
        return learned_profile

    def _weather_bucket_series(self, df: pd.DataFrame) -> pd.Series:
        """Build coarse weather-condition buckets for adaptive correction.

        Args:
            df: Dataframe containing weather columns.

        Returns:
            Series of string bucket labels aligned with ``df``.
        """
        cloud_cover = df.get("cloud_cover", pd.Series(0.0, index=df.index)).fillna(0.0)
        cloud_bucket = pd.cut(
            cloud_cover,
            bins=[-0.1, 25.0, 60.0, 85.0, 100.0],
            labels=["clear", "mixed", "cloudy", "overcast"],
        ).astype("string")

        wet_columns = ["precipitation", "rain", "showers"]
        wet_signal = pd.Series(0.0, index=df.index)
        for column in wet_columns:
            if column in df.columns:
                wet_signal = wet_signal.add(df[column].fillna(0.0), fill_value=0.0)

        snow_signal = pd.Series(0.0, index=df.index)
        for column in ["snow_depth", "snowfall", "snowfall_height"]:
            if column in df.columns:
                snow_signal = snow_signal.add(df[column].fillna(0.0), fill_value=0.0)

        weather_state = pd.Series("dry", index=df.index, dtype="string")
        weather_state = weather_state.mask(wet_signal > 0.1, "wet")
        weather_state = weather_state.mask(snow_signal > 0.01, "snow")

        if {"direct_radiation", "diffuse_radiation"}.issubset(df.columns):
            direct = df["direct_radiation"].fillna(0.0).clip(lower=0.0)
            diffuse = df["diffuse_radiation"].fillna(0.0).clip(lower=0.0)
            ratio = direct / (direct + diffuse).replace(0.0, pd.NA)
            radiation_bucket = pd.Series("unknown", index=df.index, dtype="string")
            radiation_bucket = radiation_bucket.mask(ratio < 0.35, "diffuse")
            radiation_bucket = radiation_bucket.mask(
                (ratio >= 0.35) & (ratio < 0.7), "mixed"
            )
            radiation_bucket = radiation_bucket.mask(ratio >= 0.7, "direct")
        else:
            radiation_bucket = pd.Series("unknown", index=df.index, dtype="string")

        return cloud_bucket + "|" + weather_state + "|" + radiation_bucket

    def _energy_limit_from_efficiency(
        self, future: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Calculate interval energy limit from learned PV efficiency.

        Args:
            future: Weather dataframe aligned with forecast rows.

        Returns:
            Energy limit per interval, or None when efficiency is unavailable.
        """
        base_efficiency_factor = self.config.efficiency_factor
        if base_efficiency_factor is None:
            return None

        minute_of_day = future["ds"].dt.hour * 60 + future["ds"].dt.minute
        efficiency_profile = self.config.efficiency_profile
        if efficiency_profile:
            efficiency_factor = minute_of_day.map(
                lambda minute: efficiency_profile.get(
                    int(minute), base_efficiency_factor
                )
            )
        else:
            efficiency_factor = pd.Series(base_efficiency_factor, index=future.index)

        power_limit = future["shortwave_radiation"] * (
            efficiency_factor * self.config.efficiency_cap_headroom
        )
        interval_hours = self.config.interval_minutes / 60.0
        return cast(pd.Series, power_limit * interval_hours)

    def _curtailment_limit_series(
        self,
        future: pd.DataFrame,
        dynamic_export_limits: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Build per-interval optional curtailment limit in kWh.

        Args:
            future: Forecast weather dataframe with ``ds`` timestamps.
            dynamic_export_limits: Optional timestamped export limits with ``ds``
                and either ``export_limit_kw`` or ``limit_kw``.

        Returns:
            Series containing interval kWh limits, or infinity when unrestricted.

        Raises:
            ValueError: If dynamic limits are missing required columns.
        """
        limit_kw = pd.Series(float("inf"), index=future.index)
        static_limits = [
            value
            for value in [
                self.config.inverter_ac_limit_kw,
                self.config.grid_export_limit_kw,
            ]
            if value is not None
        ]
        if static_limits:
            limit_kw = pd.Series(min(static_limits), index=future.index)

        if dynamic_export_limits is not None:
            dynamic = dynamic_export_limits.copy()
            limit_column = (
                "export_limit_kw"
                if "export_limit_kw" in dynamic.columns
                else "limit_kw"
            )
            if "ds" not in dynamic.columns or limit_column not in dynamic.columns:
                raise ValueError(
                    "dynamic_export_limits must contain 'ds' and "
                    "'export_limit_kw' or 'limit_kw' columns"
                )

            dynamic["ds"] = pd.to_datetime(dynamic["ds"], utc=True)
            dynamic = dynamic.sort_values("ds")[["ds", limit_column]]
            aligned = pd.merge_asof(
                future[["ds"]].sort_values("ds"),
                dynamic,
                on="ds",
                direction="backward",
            ).sort_index()
            dynamic_kw = aligned[limit_column].reindex(future.index)
            dynamic_kw = pd.to_numeric(dynamic_kw, errors="coerce")
            limit_kw = pd.concat([limit_kw, dynamic_kw], axis=1).min(axis=1)

        interval_hours = self.config.interval_minutes / 60.0
        return limit_kw * interval_hours

    def _calibrate_weather_correction(
        self, prepared_df: pd.DataFrame
    ) -> Optional[dict[str, Any]]:
        """Learn conservative weather correction metadata.

        Training is allowed without archived production forecasts. When forecast
        archive rows exist, calibration compares those forecast snapshots with
        later actual production. Otherwise it falls back to historical/archive
        weather and a learned efficiency baseline.

        Args:
            prepared_df: Prepared training dataframe with actual production and
                weather columns.

        Returns:
            Calibration metadata, or None when insufficient data is available.
        """
        if not self.config.adaptive_weather_correction:
            return None
        if self.config.forecast_type != "solar":
            return None

        actual = prepared_df.copy()
        actual["ds"] = pd.to_datetime(actual["ds"], utc=True)
        forecast_model = self._forecast_model_id()
        archive = self.storage.load_forecast_archive(
            self.config.model_id, forecast_model
        )

        if archive is not None and "pre_weather_correction_yhat" in archive.columns:
            archive["ds"] = pd.to_datetime(archive["ds"], utc=True)
            calibration_df = pd.merge(
                actual[["ds", "y"]],
                archive,
                on="ds",
                how="inner",
                suffixes=("_actual", ""),
            )
            baseline = calibration_df["pre_weather_correction_yhat"]
            source = "forecast_archive"
        else:
            energy_limit = self._energy_limit_from_efficiency(actual)
            if energy_limit is None:
                return None
            calibration_df = actual.copy()
            baseline = energy_limit.reindex(calibration_df.index)
            source = "historical_archive_weather"

        ratio = calibration_df["y"] / baseline.replace(0.0, pd.NA)
        ratio = pd.to_numeric(ratio, errors="coerce")
        calibration_df = calibration_df.assign(weather_ratio=ratio)
        calibration_df = calibration_df[
            (calibration_df["weather_ratio"] > 0)
            & calibration_df["weather_ratio"].notna()
            & (baseline > 0.01)
        ].copy()

        if "applied_curtailment_limit_kwh" in calibration_df.columns:
            limit = calibration_df["applied_curtailment_limit_kwh"]
            calibration_df = calibration_df[
                limit.isna() | (calibration_df["y"] < limit * 0.98)
            ]

        if len(calibration_df) < self.config.min_weather_correction_samples:
            return None

        lower = self.config.min_weather_correction_multiplier
        upper = self.config.max_weather_correction_multiplier
        global_multiplier = float(calibration_df["weather_ratio"].median())
        global_multiplier = min(max(global_multiplier, lower), upper)

        calibration_df["weather_bucket"] = self._weather_bucket_series(calibration_df)
        buckets: dict[str, dict[str, float | int]] = {}
        min_samples = self.config.min_weather_correction_samples
        for bucket, group in calibration_df.groupby("weather_bucket", dropna=True):
            sample_count = int(len(group))
            bucket_multiplier = float(group["weather_ratio"].median())
            bucket_multiplier = min(max(bucket_multiplier, lower), upper)
            confidence = min(1.0, sample_count / (min_samples * 4.0))
            shrunk_multiplier = 1.0 + confidence * (bucket_multiplier - 1.0)
            buckets[str(bucket)] = {
                "multiplier": float(shrunk_multiplier),
                "samples": sample_count,
            }

        metadata: dict[str, Any] = {
            "source": source,
            "forecast_model": forecast_model,
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "default_multiplier": global_multiplier,
            "min_samples": min_samples,
            "buckets": buckets,
            "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        self.storage.save_weather_calibration(
            self.config.model_id, forecast_model, metadata
        )
        return metadata

    def _load_weather_correction(self) -> None:
        """Load adaptive weather correction metadata when available.

        Returns:
            None.
        """
        if self.config.weather_correction is not None:
            return
        calibration = self.storage.load_weather_calibration(
            self.config.model_id, self._forecast_model_id()
        )
        if calibration is not None:
            self.config.weather_correction = calibration

    def _apply_adaptive_weather_correction(
        self, forecast: pd.DataFrame, future: pd.DataFrame
    ) -> pd.Series:
        """Apply persisted adaptive weather correction to forecast columns.

        Args:
            forecast: Forecast dataframe to mutate.
            future: Weather dataframe aligned with ``forecast``.

        Returns:
            Applied multiplier series.
        """
        self._load_weather_correction()
        correction = self.config.weather_correction
        multipliers = pd.Series(1.0, index=forecast.index)
        if not self.config.adaptive_weather_correction or not correction:
            return multipliers

        buckets = correction.get("buckets", {})
        default_multiplier = float(correction.get("default_multiplier", 1.0))
        weather_buckets = self._weather_bucket_series(future)
        multipliers = weather_buckets.map(
            lambda bucket: buckets.get(str(bucket), {}).get(
                "multiplier", default_multiplier
            )
        )
        multipliers = pd.to_numeric(multipliers, errors="coerce").fillna(1.0)

        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] *= multipliers.values
        return cast(pd.Series, multipliers)

    def _apply_curtailment(
        self,
        forecast: pd.DataFrame,
        future: pd.DataFrame,
        dynamic_export_limits: Optional[pd.DataFrame] = None,
    ) -> None:
        """Apply optional inverter/export limits to delivered forecast columns.

        Args:
            forecast: Forecast dataframe to mutate.
            future: Weather dataframe aligned with forecast rows.
            dynamic_export_limits: Optional time-varying export limits.

        Returns:
            None.
        """
        limit = self._curtailment_limit_series(future, dynamic_export_limits)
        finite_limit = limit.replace(float("inf"), pd.NA)
        forecast["applied_curtailment_limit_kwh"] = finite_limit
        forecast["pre_curtailment_yhat"] = forecast["yhat"]

        clipped_yhat = forecast["yhat"].combine(limit, min)
        forecast["curtailed_kwh"] = (forecast["yhat"] - clipped_yhat).clip(lower=0)
        forecast["curtailment_active"] = forecast["curtailed_kwh"] > 0
        forecast["yhat"] = clipped_yhat

        for col in ["yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].combine(limit, min)

    def _save_forecast_archive(
        self, forecast: pd.DataFrame, future: pd.DataFrame
    ) -> None:
        """Persist forecast outputs and weather inputs for future calibration.

        Args:
            forecast: Forecast dataframe after post-processing.
            future: Weather dataframe aligned with forecast rows.

        Returns:
            None.
        """
        weather_columns = [column for column in future.columns if column != "ds"]
        output_columns = [
            "ds",
            "pre_weather_correction_yhat",
            "weather_corrected_yhat",
            "pre_curtailment_yhat",
            "yhat",
            "applied_curtailment_limit_kwh",
            "curtailed_kwh",
            "curtailment_active",
            "weather_correction_multiplier",
        ]
        archive = forecast[[c for c in output_columns if c in forecast.columns]].copy()
        archive["ds"] = pd.to_datetime(archive["ds"], utc=True)
        weather_archive = future[["ds", *weather_columns]].copy()
        weather_archive["ds"] = pd.to_datetime(weather_archive["ds"], utc=True)
        archive = pd.merge(archive, weather_archive, on="ds", how="left")
        archive["forecast_model"] = self._forecast_model_id()
        self.storage.save_forecast_archive(
            self.config.model_id, self._forecast_model_id(), archive
        )

    def train(self, history_df: pd.DataFrame, force: bool = False):
        """
        Trains the Prophet model using the provided history.
        """
        if not force:
            if self._model:
                return

            loaded = self.storage.load_model(self.config.model_id)
            if loaded:
                self._model, metadata = loaded
                self._apply_model_metadata(metadata)
                logger.info(f"Loaded existing model {self.config.model_id}")
                return

        from .utils import normalize_to_instant_kwh

        history_df = normalize_to_instant_kwh(
            history_df,
            category=self.config.data_category.value,
            unit=self.config.unit.value,
            target_interval_min=self.config.interval_minutes,
        )

        df = self._prepare_training_data(history_df)

        # Calibrate physics-informed efficiency limit
        self.config.efficiency_factor = self.calibrate_efficiency(df)
        self.config.efficiency_profile = self.calibrate_efficiency_profile(df)
        self.config.weather_correction = self._calibrate_weather_correction(df)

        m = Prophet(
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=0.8,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
        )

        if self.config.forecast_type == "solar":
            m.add_regressor("temperature_2m")
            m.add_regressor("rolling_cloud_cover")
            m.add_regressor("shortwave_radiation")
            m.add_regressor("clear_sky_ghi")
        elif self.config.forecast_type == "consumption" and self.config.heat_pump_mode:
            m.add_regressor("temperature_2m")

        logger.info(f"Training Prophet model for {self.config.forecast_type}...")
        df_prophet = df.copy()
        df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
        m.fit(df_prophet)

        self.storage.save_model(
            m, self.config.model_id, metadata=self._build_model_metadata()
        )
        self.storage.save_training_data(self.config.model_id, df)
        self._model = m

    def tune_model(self, history_df: pd.DataFrame, days: int = 30):
        """
        Find optimal hyperparameters using cross-validation.
        """
        logger.info(f"Tuning model hyperparameters using {days} days of history...")
        df = self._prepare_training_data(history_df)

        param_grid = {
            "changepoint_prior_scale": [0.001, 0.05, 0.5],
            "seasonality_prior_scale": [0.01, 1.0, 10.0],
        }

        best_params = {}
        min_rmse = float("inf")

        for cps in param_grid["changepoint_prior_scale"]:
            for sps in param_grid["seasonality_prior_scale"]:
                m = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                )
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

                df_prophet = df.copy()
                df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
                m.fit(df_prophet)

                df_cv = cross_validation(
                    m, initial=f"{days // 2} days", period="5 days", horizon="5 days"
                )
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmse = float(df_p["rmse"].values[0])

                if rmse < min_rmse:
                    min_rmse = rmse
                    best_params = {"cps": cps, "sps": sps}

        logger.info(f"Best params found: {best_params} with RMSE {min_rmse}")
        self.config.changepoint_prior_scale = best_params["cps"]
        self.config.seasonality_prior_scale = best_params["sps"]

    def _extract_stan_init(self, model: Prophet) -> dict[str, Any]:
        """Extract Stan params from a fitted Prophet model for warm start."""
        import numpy as np

        def _scalar(arr: Any) -> float:
            # Prophet/cmdstanpy may return 1-D or 2-D arrays depending on version.
            # Flatten to 1-D and take the first element before converting to float.
            return float(np.asarray(arr).ravel()[0])

        init: dict[str, Any] = {
            "k": _scalar(model.params["k"]),
            "m": _scalar(model.params["m"]),
            "sigma_obs": _scalar(model.params["sigma_obs"]),
            "delta": np.asarray(model.params["delta"]).ravel(),
        }
        if "beta" in model.params and np.asarray(model.params["beta"]).size > 0:
            init["beta"] = np.asarray(model.params["beta"]).ravel()
        return init

    def update(
        self,
        new_data_df: pd.DataFrame,
        max_history_days: Optional[int] = None,
    ) -> None:
        """
        Incrementally updates the model with new observations.

        Loads the stored cumulative training data, appends the new data
        (normalized + weather-merged), optionally trims to a rolling window,
        recalibrates efficiency, and retrains Prophet with a warm start from
        the previous model parameters.

        If no prior training data exists on disk, falls back to a full
        train() on new_data_df.

        Args:
            new_data_df: New observations in the same format accepted by train().
            max_history_days: If set, trim the accumulated dataset to the most
                recent N days before retraining (rolling window).
        """
        from .utils import normalize_to_instant_kwh

        # Normalize and prepare new data (fetches weather only for new date range)
        new_norm = normalize_to_instant_kwh(
            new_data_df,
            category=self.config.data_category.value,
            unit=self.config.unit.value,
            target_interval_min=self.config.interval_minutes,
        )
        new_prepared = self._prepare_training_data(new_norm)

        # Load existing cumulative prepared data
        existing = self.storage.load_training_data(self.config.model_id)

        if existing is None:
            logger.info(
                f"No prior training data for {self.config.model_id} — "
                "running full initial train."
            )
            self.train(new_data_df, force=True)
            return

        # Merge: existing + new, deduplicate on ds (new data wins)
        existing["ds"] = pd.to_datetime(existing["ds"], utc=True)
        new_prepared["ds"] = pd.to_datetime(new_prepared["ds"], utc=True)

        combined = (
            pd.concat([existing, new_prepared], ignore_index=True)
            .drop_duplicates(subset=["ds"], keep="last")
            .sort_values("ds")
            .reset_index(drop=True)
        )

        if max_history_days is not None:
            cutoff = combined["ds"].max() - pd.Timedelta(days=max_history_days)
            combined = combined[combined["ds"] >= cutoff].reset_index(drop=True)

        # Recalibrate efficiency on full accumulated dataset
        self.config.efficiency_factor = self.calibrate_efficiency(combined)
        self.config.efficiency_profile = self.calibrate_efficiency_profile(combined)
        self.config.weather_correction = self._calibrate_weather_correction(combined)

        # Resolve prior model for warm start
        prior_model = self._model
        if prior_model is None:
            loaded = self.storage.load_model(self.config.model_id)
            if loaded:
                prior_model, _ = loaded

        stan_init: dict[str, Any] | str = (
            self._extract_stan_init(prior_model)
            if prior_model is not None
            else "random"
        )

        # Build and fit Prophet
        m = Prophet(
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=0.8,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
        )

        if self.config.forecast_type == "solar":
            m.add_regressor("temperature_2m")
            m.add_regressor("rolling_cloud_cover")
            m.add_regressor("shortwave_radiation")
            m.add_regressor("clear_sky_ghi")
        elif self.config.forecast_type == "consumption" and self.config.heat_pump_mode:
            m.add_regressor("temperature_2m")

        df_prophet = combined.copy()
        df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)

        logger.info(
            f"Incremental retraining {self.config.model_id} on "
            f"{len(combined)} rows (warm_start={prior_model is not None})..."
        )
        m.fit(df_prophet, init=stan_init)

        self.storage.save_model(
            m, self.config.model_id, metadata=self._build_model_metadata()
        )
        self.storage.save_training_data(self.config.model_id, combined)
        self._model = m

    def predict(
        self,
        days: int = 7,
        dynamic_export_limits: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generates forecast for the next 'days' days with physics-informed constraints.
        """
        if self._model is None:
            loaded = self.storage.load_model(self.config.model_id)
            if not loaded:
                raise RuntimeError(f"Model {self.config.model_id} not found.")
            self._model, metadata = loaded
            self._apply_model_metadata(metadata)

        m = self._model
        weather_forecast = self.weather_client.fetch_forecast(days=days)
        weather_forecast = self.weather_client.resample_weather(
            weather_forecast, self.config.interval_minutes
        )

        future = pd.DataFrame({"ds": weather_forecast["ds"]})
        future = pd.merge(future, weather_forecast, on="ds", how="left")

        weather_cols = [
            "temperature_2m",
            "cloud_cover",
            "shortwave_radiation",
            "snow_depth",
            "snowfall",
        ]
        available_weather = [c for c in weather_cols if c in future.columns]
        future[available_weather] = (
            future[available_weather].interpolate(method="linear").bfill().ffill()
        )

        future = self._prepare_features(future)
        future_prophet = future.copy()
        future_prophet["ds"] = future_prophet["ds"].dt.tz_localize(None)

        forecast = m.predict(future_prophet)

        # 1. Solar Production Constraints
        if self.config.forecast_type == "solar":
            logger.info("Applying physical solar constraints...")

            # A. Snow Masking (The "Winter Killer")
            # Hard Cap: snow_depth > 5cm (0.05m) -> 0% production
            # Soft Penalty: snow_depth > 1cm (0.01m) -> 50% production
            snow_mask_hard = future["snow_depth"] > 0.05
            snow_mask_soft = (future["snow_depth"] > 0.01) & (
                future["snow_depth"] <= 0.05
            )

            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                # Apply Snow Constraints
                forecast.loc[snow_mask_soft.values, col] *= 0.5
                forecast.loc[snow_mask_hard.values, col] = 0.0

            # C. Efficiency Cap
            energy_limit = self._energy_limit_from_efficiency(future)
            if energy_limit is not None:
                energy_limit.index = forecast.index

                for col in ["yhat", "yhat_lower", "yhat_upper"]:
                    forecast[col] = forecast[col].combine(energy_limit, min)

            # D. Night Mask
            logger.info("Applying night mask for solar forecast...")
            elevations = calculate_solar_elevation(
                self.config.latitude, self.config.longitude, forecast["ds"]
            )
            forecast.loc[elevations < 0, ["yhat", "yhat_lower", "yhat_upper"]] = 0

            forecast["pre_weather_correction_yhat"] = forecast["yhat"]
            forecast["weather_correction_multiplier"] = (
                self._apply_adaptive_weather_correction(forecast, future)
            )
            forecast["weather_corrected_yhat"] = forecast["yhat"]
            self._apply_curtailment(forecast, future, dynamic_export_limits)

        # Clip all forecasts to 0
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(lower=0)

        if self.config.forecast_type == "solar":
            self._save_forecast_archive(forecast, future)

        return cast(pd.DataFrame, forecast)

    def get_prediction_intervals(self, days: int = 7) -> List[PredictionInterval]:
        """
        Returns prediction intervals for EMS.
        """
        forecast = self.predict(days=days)
        intervals = []
        for row in forecast.itertuples(index=False):
            intervals.append(
                PredictionInterval(
                    timestamp=pd.Timestamp(row.ds),  # type: ignore
                    expected_kwh=float(row.yhat),  # type: ignore
                    lower_bound_kwh=float(row.yhat_lower),  # type: ignore
                    upper_bound_kwh=float(row.yhat_upper),  # type: ignore
                )
            )
        return intervals

    def get_surplus_probability(
        self, threshold_kwh: float, days: int = 7
    ) -> pd.DataFrame:
        """
        Returns probability of exceeding threshold_kwh.
        """
        forecast = self.predict(days=days)
        sigma = (forecast["yhat_upper"] - forecast["yhat_lower"]) / (2 * 1.28)
        sigma = sigma.replace(0, 1e-9)

        from scipy.stats import norm

        z_score = (threshold_kwh - forecast["yhat"]) / sigma
        prob_exceed = 1 - norm.cdf(z_score)

        return pd.DataFrame({"ds": forecast["ds"], "surplus_prob": prob_exceed})
