"""Public ML forecasting API."""

from datetime import date, datetime
from numbers import Real
from typing import Any, Optional

import pandas as pd

from kpower_forecast.core import PredictionInterval
from kpower_forecast.ml.backends import create_backend
from kpower_forecast.ml.bias_correction import WeatherBiasCorrector
from kpower_forecast.ml.config import KPowerMLConfig, MLBackendType, MLForecastType
from kpower_forecast.ml.conformal import SplitConformalCalibrator
from kpower_forecast.ml.features import MLFeatureBuilder
from kpower_forecast.ml.storage import MLModelManifest, MLModelStorage
from kpower_forecast.utils import calculate_solar_elevation, normalize_to_instant_kwh
from kpower_forecast.weather_client import WeatherClient, WeatherConfig

DYNAMIC_EXPORT_LIMIT_COLUMNS: tuple[str, ...] = (
    "grid_export_limit_kw",
    "export_limit_kw",
    "curtailment_limit_kw",
    "limit_kw",
)


class KPowerMLForecast:
    """Train and serve optional ML forecasts for energy time series."""

    def __init__(
        self,
        model_id: str,
        latitude: float,
        longitude: float,
        storage_path: str = "./data",
        interval_minutes: int = 15,
        forecast_type: MLForecastType = MLForecastType.SOLAR,
        backend: MLBackendType = MLBackendType.NIXTLA_HYBRID,
        weather_config: Optional[WeatherConfig] = None,
        **config_overrides: Any,
    ):
        self.config = KPowerMLConfig(
            model_id=model_id,
            latitude=latitude,
            longitude=longitude,
            storage_path=storage_path,
            interval_minutes=interval_minutes,
            forecast_type=forecast_type,
            backend=backend,
            **config_overrides,
        )
        self.weather_client = WeatherClient(
            lat=self.config.latitude,
            lon=self.config.longitude,
            config=weather_config,
        )
        self.storage = MLModelStorage(self.config.storage_path, self.config.model_id)
        self.feature_builder = MLFeatureBuilder(self.config)
        self.backend = create_backend(self.config)
        self.bias_corrector = WeatherBiasCorrector(
            min_samples=self.config.min_weather_correction_samples
        )
        self.conformal = SplitConformalCalibrator(self.config.interval_levels)
        self._restore_existing_manifest()

    def train(self, history_df: pd.DataFrame, force: bool = False) -> None:
        """Train the configured ML backend.

        Args:
            history_df: Input history with ``ds`` and ``y`` columns.
            force: Retrain even when a manifest exists.

        Returns:
            None.
        """
        existing_manifest = self.storage.load_manifest()
        if not force and existing_manifest is not None:
            self._restore_from_manifest(existing_manifest)
            return

        normalized = normalize_to_instant_kwh(
            history_df,
            category=self.config.data_category.value,
            unit=self.config.unit.value,
            target_interval_min=self.config.interval_minutes,
        )
        prepared = self._prepare_training_data(normalized)
        train_frame, calibration_frame = self._chronological_split(prepared)
        train_features = self.feature_builder.build(train_frame)
        calibration_features = self.feature_builder.build(calibration_frame)

        self.bias_corrector.fit_from_historical_proxy(train_frame)
        self.backend.fit(train_frame, train_features, calibration_frame)
        calibration_predictions = self.backend.predict(
            calibration_features, horizon=len(calibration_frame)
        )
        self.conformal.fit(
            actual=calibration_frame["y"], predicted=calibration_predictions["yhat"]
        )

        manifest = MLModelManifest(
            model_id=self.config.model_id,
            backend_type=self.config.backend.value,
            target_type=self.config.forecast_type.value,
            interval_levels=self.config.interval_levels,
            feature_columns=self.backend.feature_schema(),
            artifact_paths=self.backend.save(self.storage.artifact_dir),
            conformal_quantiles=self.conformal.to_dict(),
            weather_bias_source=self.bias_corrector.source,
            training_start=pd.Timestamp(prepared["ds"].min()).isoformat(),
            training_end=pd.Timestamp(prepared["ds"].max()).isoformat(),
            metadata={
                "weather_bias": self.bias_corrector.to_dict(),
                "pv_limits": self._pv_limit_metadata(),
            },
        )
        self.storage.save_training_frame(prepared)
        self.storage.save_manifest(manifest)

    def predict(
        self,
        days: int = 7,
        dynamic_export_limits: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate an ML forecast for the next ``days`` days."""
        horizon = days * (24 * 60 // self.config.interval_minutes)
        weather = self.weather_client.fetch_forecast(days=days)
        weather = self.weather_client.resample_weather(
            weather, self.config.interval_minutes
        )
        weather = self.bias_corrector.apply(weather)
        features = self.feature_builder.build(weather)
        forecast = self.backend.predict(features, horizon=horizon)
        forecast = self.conformal.apply(forecast)
        if self.config.forecast_type == MLForecastType.SOLAR:
            forecast = self._apply_solar_constraints(forecast)
            forecast = self._apply_pv_curtailment(
                forecast, dynamic_export_limits=dynamic_export_limits
            )
        return forecast

    def _restore_existing_manifest(self) -> None:
        """Load persisted manifest state when available."""
        manifest = self.storage.load_manifest()
        if manifest is None:
            return
        self._restore_from_manifest(manifest)

    def _restore_from_manifest(self, manifest: MLModelManifest) -> None:
        """Restore backend, interval, and correction state from a manifest."""
        if manifest.backend_type != self.config.backend.value:
            raise ValueError(
                "stored ML backend does not match configured backend: "
                f"{manifest.backend_type} != {self.config.backend.value}"
            )
        if manifest.target_type != self.config.forecast_type.value:
            raise ValueError(
                "stored ML target does not match configured target: "
                f"{manifest.target_type} != {self.config.forecast_type.value}"
            )

        self.backend.load(self.storage.artifact_dir)
        self.conformal = SplitConformalCalibrator.from_dict(
            manifest.interval_levels, manifest.conformal_quantiles
        )
        weather_bias = manifest.metadata.get("weather_bias")
        if isinstance(weather_bias, dict):
            self.bias_corrector.load_dict(weather_bias)

    def get_prediction_intervals(
        self, days: int = 7, level: int = 90
    ) -> list[PredictionInterval]:
        """Return EMS-compatible prediction intervals for an ML forecast."""
        forecast = self.predict(days=days)
        lower_column = f"yhat_lower_{level}"
        upper_column = f"yhat_upper_{level}"
        if lower_column not in forecast.columns or upper_column not in forecast.columns:
            raise ValueError(f"interval level {level} is not available")

        return [
            PredictionInterval(
                timestamp=self._coerce_timestamp(row.ds),
                expected_kwh=self._coerce_float(row.yhat),
                lower_bound_kwh=self._coerce_float(getattr(row, lower_column)),
                upper_bound_kwh=self._coerce_float(getattr(row, upper_column)),
            )
            for row in forecast.itertuples(index=False)
        ]

    def _coerce_timestamp(self, value: object) -> pd.Timestamp:
        """Coerce an arbitrary value to a timezone-aware pandas timestamp."""
        if isinstance(value, pd.Timestamp):
            timestamp = pd.to_datetime(value, utc=True)
        elif isinstance(value, str):
            timestamp = pd.to_datetime(value, utc=True)
        elif isinstance(value, Real):
            timestamp = pd.to_datetime(float(value), utc=True)
        elif isinstance(value, datetime):
            timestamp = pd.to_datetime(value, utc=True)
        elif isinstance(value, date):
            timestamp = pd.to_datetime(value, utc=True)
        else:
            raise ValueError(f"invalid timestamp value: {value!r}")

        if isinstance(timestamp, pd.Timestamp):
            return timestamp
        raise ValueError(f"invalid timestamp value: {value!r}")

    def _coerce_float(self, value: object) -> float:
        """Coerce a scalar value to a float."""
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, str):
            return float(value)
        numeric = pd.to_numeric([value], errors="coerce")
        if pd.isna(numeric[0]):
            raise ValueError(f"invalid numeric value: {value!r}")
        return float(numeric[0])

    def _prepare_training_data(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Merge normalized history with historical weather and ML features."""
        if "ds" not in history_df.columns or "y" not in history_df.columns:
            raise ValueError("history_df must contain 'ds' and 'y' columns")
        history = history_df.copy()
        history["ds"] = pd.to_datetime(history["ds"], utc=True)
        start_date = history["ds"].min().date()
        end_date = history["ds"].max().date()
        weather = self.weather_client.fetch_historical(start_date, end_date)
        weather = self.weather_client.resample_weather(
            weather, self.config.interval_minutes
        )
        prepared = pd.merge(history, weather, on="ds", how="left")
        weather_columns = [
            column for column in prepared.columns if column not in {"ds", "y"}
        ]
        prepared[weather_columns] = (
            prepared[weather_columns].interpolate().bfill().ffill()
        )
        return prepared.dropna(subset=["y"]).reset_index(drop=True)

    def _chronological_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a dataframe into training and calibration windows."""
        if len(df) < 4:
            raise ValueError("at least four rows are required for ML training")
        calibration_size = max(1, int(len(df) * self.config.calibration_fraction))
        if len(df) - calibration_size < 2:
            calibration_size = 1
        return df.iloc[:-calibration_size].copy(), df.iloc[-calibration_size:].copy()

    def _apply_solar_constraints(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Apply basic non-negative and night-time constraints for solar forecasts."""
        output = forecast.copy()
        output["ds"] = pd.to_datetime(output["ds"], utc=True)
        elevations = calculate_solar_elevation(
            self.config.latitude,
            self.config.longitude,
            list(output["ds"].dt.to_pydatetime()),
        )
        forecast_columns = [
            column for column in output.columns if column.startswith("yhat")
        ]
        output.loc[elevations < 0, forecast_columns] = 0.0
        output[forecast_columns] = output[forecast_columns].clip(lower=0.0)
        return output

    def _apply_pv_curtailment(
        self,
        forecast: pd.DataFrame,
        dynamic_export_limits: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Apply inverter AC and export curtailment caps to solar energy forecasts."""
        output = forecast.copy()
        forecast_columns = [
            column for column in output.columns if column.startswith("yhat")
        ]
        if not forecast_columns:
            return output

        cap_kwh = self._static_curtailment_cap_kwh(len(output))
        if dynamic_export_limits is not None:
            dynamic_cap = self._dynamic_export_cap_kwh(output, dynamic_export_limits)
            cap_kwh = (
                dynamic_cap if cap_kwh is None else cap_kwh.combine(dynamic_cap, min)
            )

        if cap_kwh is None:
            return output

        cap_kwh = cap_kwh.clip(lower=0.0).reset_index(drop=True)
        for column in forecast_columns:
            clipped = output[column].reset_index(drop=True).clip(upper=cap_kwh)
            output.loc[:, column] = clipped.to_numpy()
        return output

    def _static_curtailment_cap_kwh(self, rows: int) -> Optional[pd.Series]:
        """Return the static inverter/export cap in interval kWh when configured."""
        limits_kw = [
            limit
            for limit in [
                self.config.inverter_ac_limit_kw,
                self.config.grid_export_limit_kw,
            ]
            if limit is not None
        ]
        if not limits_kw:
            return None

        interval_hours = self.config.interval_minutes / 60.0
        cap = min(limits_kw) * interval_hours
        return pd.Series([cap] * rows, dtype="float64")

    def _dynamic_export_cap_kwh(
        self, forecast: pd.DataFrame, dynamic_export_limits: pd.DataFrame
    ) -> pd.Series:
        """Align dynamic export limits to the forecast grid and convert kW to kWh."""
        if "ds" not in dynamic_export_limits.columns:
            raise ValueError("dynamic_export_limits must contain a 'ds' column")

        limit_column = next(
            (
                column
                for column in DYNAMIC_EXPORT_LIMIT_COLUMNS
                if column in dynamic_export_limits.columns
            ),
            None,
        )
        if limit_column is None:
            expected = ", ".join(DYNAMIC_EXPORT_LIMIT_COLUMNS)
            raise ValueError(
                f"dynamic_export_limits must contain one of these columns: {expected}"
            )

        limits = dynamic_export_limits[["ds", limit_column]].copy()
        limits["ds"] = pd.to_datetime(limits["ds"], utc=True)
        limits[limit_column] = pd.to_numeric(limits[limit_column], errors="coerce")
        limits = limits.dropna(subset=[limit_column]).sort_values("ds")
        if limits.empty:
            raise ValueError("dynamic_export_limits contains no usable limit values")

        forecast_grid = forecast[["ds"]].copy()
        forecast_grid["_order"] = range(len(forecast_grid))
        forecast_grid["ds"] = pd.to_datetime(forecast_grid["ds"], utc=True)
        aligned = pd.merge_asof(
            forecast_grid.sort_values("ds"),
            limits,
            on="ds",
            direction="backward",
        )
        aligned[limit_column] = aligned[limit_column].ffill().bfill()
        aligned = aligned.sort_values("_order")
        interval_hours = self.config.interval_minutes / 60.0
        return aligned[limit_column].reset_index(drop=True) * interval_hours

    def _pv_limit_metadata(self) -> dict[str, Optional[float]]:
        """Return configured static PV limit metadata for the manifest."""
        return {
            "inverter_ac_limit_kw": self.config.inverter_ac_limit_kw,
            "grid_export_limit_kw": self.config.grid_export_limit_kw,
        }
