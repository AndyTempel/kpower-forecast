import logging
from enum import Enum
from typing import List, Literal, Optional, cast

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import ModelStorage
from .utils import calculate_solar_elevation, get_clear_sky_ghi
from .weather_client import WeatherClient

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
    max_efficiency_factor: float = 0.025  # Safety cap (approx 25kWp system)
    cloud_impact: float = 0.35  # Linear damping at 100% cloud cover

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
        data_category: DataCategory = DataCategory.INSTANT_ENERGY,
        unit: MeasurementUnit = MeasurementUnit.KWH,
        heat_pump_mode: bool = False,
        cloud_impact: float = 0.35,
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
            cloud_impact=cloud_impact,
        )

        self.weather_client = WeatherClient(
            lat=self.config.latitude, lon=self.config.longitude
        )
        self.storage = ModelStorage(storage_path=self.config.storage_path)
        self._model: Optional[Prophet] = None

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

        # Filter for clear sky (<15% cloud cover) and non-zero irradiance
        # NEW: Filter out snowy days (snow_depth > 0.01m)
        mask = (df["cloud_cover"] < 15) & (df["shortwave_radiation"] > 50)
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

        # Hard safety cap
        if factor > self.config.max_efficiency_factor:
            logger.warning(
                f"Calibrated factor {factor:.6f} exceeds safety limit "
                f"{self.config.max_efficiency_factor}. Clamping."
            )
            factor = self.config.max_efficiency_factor

        logger.info(f"Calibrated efficiency factor: {factor:.6f} kW per W/m2")
        return factor

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
                self.config.efficiency_factor = metadata.get("efficiency_factor")
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

        metadata = {"efficiency_factor": self.config.efficiency_factor}
        self.storage.save_model(m, self.config.model_id, metadata=metadata)
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

    def predict(self, days: int = 7) -> pd.DataFrame:
        """
        Generates forecast for the next 'days' days with physics-informed constraints.
        """
        if self._model is None:
            loaded = self.storage.load_model(self.config.model_id)
            if not loaded:
                raise RuntimeError(f"Model {self.config.model_id} not found.")
            self._model, metadata = loaded
            self.config.efficiency_factor = metadata.get("efficiency_factor")

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
            logger.info("Applying physical constraints (Snow & Clouds)...")

            # A. Snow Masking (The "Winter Killer")
            # Hard Cap: snow_depth > 5cm (0.05m) -> 0% production
            # Soft Penalty: snow_depth > 1cm (0.01m) -> 50% production
            snow_mask_hard = future["snow_depth"] > 0.05
            snow_mask_soft = (future["snow_depth"] > 0.01) & (
                future["snow_depth"] <= 0.05
            )

            # B. Linear Cloud Damping (Diffuse Light Physics)
            # Factor = 1.0 - (cloud_cover_fraction * cloud_impact)
            # This allows diffuse light to pass through even at 100% clouds.
            cloud_damping = (
                1.0 - (future["cloud_cover"] / 100.0) * self.config.cloud_impact
            )

            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                # Apply Cloud Damping
                forecast[col] *= cloud_damping.values
                # Apply Snow Constraints
                forecast.loc[snow_mask_soft.values, col] *= 0.5
                forecast.loc[snow_mask_hard.values, col] = 0.0

            # C. Efficiency Cap
            if self.config.efficiency_factor:
                power_limit = (
                    future["shortwave_radiation"] * self.config.efficiency_factor
                )
                interval_hours = self.config.interval_minutes / 60.0
                energy_limit = power_limit * interval_hours
                energy_limit.index = forecast.index

                for col in ["yhat", "yhat_lower", "yhat_upper"]:
                    forecast[col] = forecast[col].combine(energy_limit, min)

            # D. Night Mask
            logger.info("Applying night mask for solar forecast...")
            elevations = calculate_solar_elevation(
                self.config.latitude, self.config.longitude, forecast["ds"]
            )
            forecast.loc[elevations < 0, ["yhat", "yhat_lower", "yhat_upper"]] = 0

        # Clip all forecasts to 0
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
        """
        forecast = self.predict(days=days)
        sigma = (forecast["yhat_upper"] - forecast["yhat_lower"]) / (2 * 1.28)
        sigma = sigma.replace(0, 1e-9)

        from scipy.stats import norm

        z_score = (threshold_kwh - forecast["yhat"]) / sigma
        prob_exceed = 1 - norm.cdf(z_score)

        return pd.DataFrame({"ds": forecast["ds"], "surplus_prob": prob_exceed})
