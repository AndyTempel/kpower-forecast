"""Feature engineering for the optional ML forecasting flow."""

import math

import pandas as pd

from kpower_forecast.ml.config import KPowerMLConfig, MLForecastType
from kpower_forecast.utils import calculate_solar_elevation, get_clear_sky_ghi


class MLFeatureBuilder:
    """Build weather, calendar, and physics features for ML backends."""

    def __init__(self, config: KPowerMLConfig):
        self.config = config

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for a dataframe containing ``ds`` and weather columns."""
        if "ds" not in df.columns:
            raise ValueError("feature dataframe must contain a 'ds' column")

        features = df.copy()
        features["ds"] = pd.to_datetime(features["ds"], utc=True)
        ds = features["ds"]
        hour = ds.dt.hour + ds.dt.minute / 60.0
        day_of_week = ds.dt.dayofweek.astype(float)
        day_of_year = ds.dt.dayofyear.astype(float)

        features["hour_sin"] = (2.0 * math.pi * hour / 24.0).map(math.sin)
        features["hour_cos"] = (2.0 * math.pi * hour / 24.0).map(math.cos)
        features["dow_sin"] = (2.0 * math.pi * day_of_week / 7.0).map(math.sin)
        features["dow_cos"] = (2.0 * math.pi * day_of_week / 7.0).map(math.cos)
        features["doy_sin"] = (2.0 * math.pi * day_of_year / 366.0).map(math.sin)
        features["doy_cos"] = (2.0 * math.pi * day_of_year / 366.0).map(math.cos)
        features["is_weekend"] = ds.dt.dayofweek.isin([5, 6]).astype(int)
        features["is_holiday"] = self._holiday_flags(ds)

        indexed = features.set_index("ds")
        if isinstance(indexed.index, pd.DatetimeIndex):
            features["clear_sky_ghi"] = get_clear_sky_ghi(
                self.config.latitude,
                self.config.longitude,
                indexed.index,
            ).to_numpy()
        else:
            features["clear_sky_ghi"] = 0.0

        features["solar_elevation"] = calculate_solar_elevation(
            self.config.latitude,
            self.config.longitude,
            list(ds.dt.to_pydatetime()),
        )

        if "shortwave_radiation" in features.columns:
            denominator = features["clear_sky_ghi"].replace(0.0, pd.NA)
            features["clear_sky_index"] = (
                features["shortwave_radiation"] / denominator
            ).fillna(0.0)
        else:
            features["clear_sky_index"] = 0.0

        if {"wind_u_component_10m", "wind_v_component_10m"}.issubset(features.columns):
            features["wind_speed_10m"] = (
                features["wind_u_component_10m"] ** 2
                + features["wind_v_component_10m"] ** 2
            ) ** 0.5

        if "temperature_2m" in features.columns:
            features["heating_degree"] = (18.0 - features["temperature_2m"]).clip(
                lower=0.0
            )
            features["cooling_degree"] = (features["temperature_2m"] - 22.0).clip(
                lower=0.0
            )

        self._add_rolling_std(features, "temperature_2m")
        self._add_rolling_std(features, "shortwave_radiation")
        if self.config.forecast_type == MLForecastType.HVAC:
            features["hvac_temperature_delta"] = features.get(
                "heating_degree", 0.0
            ) + features.get("cooling_degree", 0.0)
            if "hvac_power_w" in features.columns:
                features["hvac_power_kw"] = (
                    pd.to_numeric(features["hvac_power_w"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                    / 1000.0
                )
            if "hvac_setpoint_c" in features.columns:
                setpoint = pd.to_numeric(features["hvac_setpoint_c"], errors="coerce")
                if "y" in features.columns:
                    indoor = pd.to_numeric(features["y"], errors="coerce")
                    features["hvac_setpoint_gap_c"] = (setpoint - indoor).fillna(0.0)
                else:
                    features["hvac_setpoint_gap_c"] = setpoint.fillna(0.0)

        return features.fillna(0.0)

    def _holiday_flags(self, ds: pd.Series) -> pd.Series:
        """Return holiday flags for configured country/subdivision."""
        if not self.config.holiday_country:
            return pd.Series(0, index=ds.index, dtype="int64")

        try:
            import holidays
        except ImportError as exc:
            raise RuntimeError(
                "holiday features require kpower-forecast[ml] dependencies"
            ) from exc

        years = sorted(set(ds.dt.year.astype(int).tolist()))
        calendar = holidays.country_holidays(
            self.config.holiday_country,
            subdiv=self.config.holiday_subdivision,
            years=years,
        )
        holiday_flags = ds.dt.date.map(lambda date_value: int(date_value in calendar))
        return pd.Series(holiday_flags.to_numpy(), index=ds.index, dtype="int64")

    def _add_rolling_std(self, features: pd.DataFrame, column: str) -> None:
        """Add 3-hour and 6-hour rolling standard deviations for ``column``."""
        if column not in features.columns:
            return

        for hours in (3, 6):
            window = max(1, int((hours * 60) / self.config.interval_minutes))
            features[f"{column}_std_{hours}h"] = (
                features[column].rolling(window=window, min_periods=1).std().fillna(0.0)
            )
