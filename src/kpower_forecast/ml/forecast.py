"""Public ML forecasting API."""

from datetime import date, datetime
from math import ceil
from numbers import Real
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from kpower_forecast import __version__
from kpower_forecast.core import PredictionInterval
from kpower_forecast.ml.alignment import (
    FORECAST_CONTRACT_VERSION,
    ForecastAlignmentError,
    prediction_origin,
    validate_timestamp_grid,
)
from kpower_forecast.ml.backends import create_backend
from kpower_forecast.ml.baselines import BASELINE_NAME, local_slot_weekday_class_median
from kpower_forecast.ml.bias_correction import WeatherBiasCorrector
from kpower_forecast.ml.config import (
    HybridStructure,
    KPowerMLConfig,
    MLBackendType,
    MLForecastType,
)
from kpower_forecast.ml.conformal import SplitConformalCalibrator
from kpower_forecast.ml.features import MLFeatureBuilder
from kpower_forecast.ml.selection import (
    BLEND_CANDIDATE,
    DEGREE_HOUR_CANDIDATE,
    ML_CANDIDATE,
    SELECTION_CANDIDATES,
    SELECTION_METRIC,
    CandidateMetrics,
    DegreeHourRegression,
    bias_eligible,
    fit_blend_weight,
    score_windows,
    select_candidate,
)
from kpower_forecast.ml.storage import MLModelManifest, MLModelStorage
from kpower_forecast.utils import calculate_solar_elevation, normalize_to_instant_kwh
from kpower_forecast.weather_client import WeatherClient, WeatherConfig

DYNAMIC_EXPORT_LIMIT_COLUMNS: tuple[str, ...] = (
    "grid_export_limit_kw",
    "export_limit_kw",
    "curtailment_limit_kw",
    "limit_kw",
)
SANITIZED_CONFORMAL_STATE_VERSION: int = 1
# The hourly archive's last value is 23:00 while the recent forecast starts at
# 00:00, so up to one hour between the two sources may be interpolated.
MAX_WEATHER_SEAM: pd.Timedelta = pd.Timedelta(hours=1)
HISTORY_POLICY_VERSION: int = 2


def bridge_short_target_gaps(
    history: pd.DataFrame, max_intervals: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Bridge short interior target gaps on a regular grid.

    Telemetry with intermittent attribution drops single rows far more often
    than it loses whole hours. Those rows would otherwise split the latest
    contiguous structural segment. Runs of at most ``max_intervals`` missing
    rows with an observation on both sides are filled linearly; longer gaps and
    leading or trailing gaps stay missing.

    Args:
        history: Grid-aligned frame with ``ds`` and ``y``; gaps are NaN rows.
        max_intervals: Longest missing run to bridge. ``0`` disables bridging.

    Returns:
        A copy of ``history`` with bridged ``y`` values and a boolean mask, on
        the same index, marking the bridged rows.

    Raises:
        ValueError: If ``max_intervals`` is negative or ``y`` is missing.
    """
    if max_intervals < 0:
        raise ValueError("max_intervals must not be negative")
    if "y" not in history.columns:
        raise ValueError("history must contain a 'y' column")
    values = pd.to_numeric(history["y"], errors="coerce")
    missing = values.isna()
    bridged = pd.Series(False, index=history.index)
    if max_intervals == 0 or not bool(missing.any()):
        return history.copy(), bridged
    run_id = missing.ne(missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("size")
    interior = values.ffill().notna() & values.bfill().notna()
    bridged = missing & interior & run_length.le(max_intervals)
    output = history.copy()
    filled = values.interpolate(method="linear", limit_area="inside")
    output.loc[bridged, "y"] = filled.loc[bridged]
    return output, bridged


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
        timezone: str = "UTC",
        preserve_gaps: bool = False,
        **config_overrides: Any,
    ):
        self.config = KPowerMLConfig(
            model_id=model_id,
            latitude=latitude,
            longitude=longitude,
            storage_path=storage_path,
            interval_minutes=interval_minutes,
            timezone=timezone,
            preserve_gaps=preserve_gaps,
            forecast_type=forecast_type,
            backend=backend,
            **config_overrides,
        )
        self.weather_client = WeatherClient(
            lat=self.config.latitude,
            lon=self.config.longitude,
            config=self._weather_config_with_default_cache(weather_config),
        )
        self.storage = MLModelStorage(self.config.storage_path, self.config.model_id)
        self.feature_builder = MLFeatureBuilder(self.config)
        self.backend = create_backend(self.config)
        self.bias_corrector = WeatherBiasCorrector(
            min_samples=self.config.min_weather_correction_samples
        )
        self.conformal = SplitConformalCalibrator(self.config.interval_levels)
        self._training_end: pd.Timestamp | None = None
        self.training_bridged_rows: int = 0
        self.selected_candidate: str = ML_CANDIDATE
        self.candidate_metrics: dict[str, dict[str, float | int]] = {}
        self.selection_reason: Optional[str] = None
        self._regression: Optional[DegreeHourRegression] = None
        self._blend_weight: Optional[float] = None
        self._restore_existing_manifest()

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

        covariates = self._history_covariates(history_df)
        # Only the target is normalized; covariates are merged separately.
        normalized = normalize_to_instant_kwh(
            history_df[["ds", "y"]],
            category=self.config.data_category.value,
            unit=self.config.unit.value,
            target_interval_min=self.config.interval_minutes,
            preserve_gaps=self.config.preserve_gaps,
        )
        complete_history = self._prepare_training_data(normalized)
        if covariates is not None:
            clash = set(self.config.known_covariates) & set(complete_history.columns)
            if clash:
                raise ValueError(
                    f"known covariates {sorted(clash)} clash with weather columns"
                )
            complete_history = complete_history.merge(covariates, on="ds", how="left")
        bridged_history, bridged_mask = bridge_short_target_gaps(
            complete_history, self.config.max_bridged_gap_intervals
        )
        bridged_times = set(
            pd.to_datetime(bridged_history.loc[bridged_mask, "ds"], utc=True)
        )
        complete_features = self.feature_builder.build(bridged_history)
        prepared, prepared_features, train_frame, calibration_frame = (
            self._gap_safe_training_split(
                bridged_history, complete_features, bridged=bridged_mask
            )
        )
        train_features = self.feature_builder.build(train_frame)
        calibration_features = self.feature_builder.build(calibration_frame)
        full_features = prepared_features
        self._training_end = pd.to_datetime(prepared["ds"], utc=True).max()

        # Bridged rows keep the structural series contiguous, but only measured
        # rows may calibrate weather bias or prediction intervals.
        train_measured = ~pd.to_datetime(train_frame["ds"], utc=True).isin(
            bridged_times
        )
        self.bias_corrector.fit_from_historical_proxy(
            train_frame.loc[train_measured].reset_index(drop=True)
        )
        self.backend.fit(train_frame, train_features, calibration_frame)
        calibration_predictions = self.backend.predict(
            calibration_features, horizon=len(calibration_frame)
        )
        calibration_predictions = self._sanitize_point_forecast(calibration_predictions)
        calibration_measured = ~pd.to_datetime(calibration_frame["ds"], utc=True).isin(
            bridged_times
        )
        calibration_actual = (
            calibration_frame["y"]
            .reset_index(drop=True)
            .where(calibration_measured.reset_index(drop=True))
        )
        conformal_actual, conformal_predicted = self._select_candidate(
            bridged_history,
            complete_features,
            bridged_mask,
            calibration_actual,
            calibration_predictions["yhat"].reset_index(drop=True),
        )
        # Prediction intervals describe the forecast that is actually served,
        # calibrated on residuals that did not take part in selecting it.
        self.conformal.fit(actual=conformal_actual, predicted=conformal_predicted)
        self.training_bridged_rows = int(bridged_mask.sum())
        self.backend.fit(prepared, full_features, calibration_frame)
        if self._regression is not None:
            prepared_measured = ~pd.to_datetime(prepared["ds"], utc=True).isin(
                bridged_times
            )
            if not self._regression.fit(prepared.loc[prepared_measured]):
                raise ValueError("degree-hour regression refit failed")

        self.storage.invalidate_manifest()
        manifest = MLModelManifest(
            contract_version=FORECAST_CONTRACT_VERSION,
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
            package_version=__version__,
            metadata={
                "weather_bias": self.bias_corrector.to_dict(),
                "pv_limits": self._pv_limit_metadata(),
                "sanitized_conformal_state_version": (
                    SANITIZED_CONFORMAL_STATE_VERSION
                ),
                "timezone": self.config.timezone,
                "history_policy_version": HISTORY_POLICY_VERSION,
                "preserve_gaps": self.config.preserve_gaps,
                "max_bridged_gap_intervals": self.config.max_bridged_gap_intervals,
                "hybrid_structure": self.config.hybrid_structure.value,
                "profile_lookback_days": self.config.profile_lookback_days,
                "profile_class_prior_days": self.config.profile_class_prior_days,
                "profile_smoothing_minutes": self.config.profile_smoothing_minutes,
                "known_covariates": list(self.config.known_covariates),
                "bridged_rows": self.training_bridged_rows,
                "candidate_selection": {
                    **_selection_settings(self.config),
                    "metric": SELECTION_METRIC,
                    "selected": self.selected_candidate,
                    "reason": self.selection_reason,
                    "candidates": self.candidate_metrics,
                    "blend_weight": self._blend_weight,
                },
                "degree_hour_regression": (
                    self._regression.to_dict() if self._regression is not None else None
                ),
            },
        )
        self.storage.save_training_frame(complete_history)
        self.storage.save_manifest(manifest)

    def predict(
        self,
        days: int = 7,
        dynamic_export_limits: Optional[pd.DataFrame] = None,
        origin: datetime | None = None,
        known_future: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate an aligned ML forecast for the next ``days`` days.

        Args:
            days: Number of complete 24-hour periods to return.
            dynamic_export_limits: Optional time-varying solar export limits.
            origin: Optional timezone-aware first returned timestamp. It must be
                interval-aligned and cannot precede the first post-training slot.
                When omitted, prediction starts at the next current interval.
            known_future: ``ds`` plus every ``known_covariates`` column for each
                model-grid row, i.e. from the first post-training slot (not
                only from ``origin``) through the returned horizon.

        Returns:
            Forecast dataframe containing exactly ``days`` of future intervals.

        Raises:
            ForecastAlignmentError: If the training cutoff, origin, weather grid,
                backend output, or returned horizon is not exactly aligned.
        """
        if days <= 0:
            raise ValueError("days must be positive")
        if self._training_end is None:
            raise ForecastAlignmentError("model training cutoff is not available")

        interval_minutes = self.config.interval_minutes
        interval = pd.Timedelta(minutes=interval_minutes)
        model_start = self._training_end + interval
        current_start = pd.Timestamp.now(tz="UTC").ceil(f"{interval_minutes}min")
        requested_start = (
            max(model_start, current_start)
            if origin is None
            else prediction_origin(origin, interval_minutes)
        )
        if requested_start < model_start:
            raise ForecastAlignmentError(
                f"prediction origin {requested_start.isoformat()} precedes first "
                f"post-training slot {model_start.isoformat()}"
            )
        offset = requested_start - model_start
        if offset % interval != pd.Timedelta(0):
            raise ForecastAlignmentError(
                "prediction origin is not reachable on the model interval grid"
            )

        returned_horizon = days * (24 * 60 // interval_minutes)
        skipped_steps = int(offset / interval)
        model_horizon = skipped_steps + returned_horizon
        requested_end = requested_start + returned_horizon * interval
        current_day_start = current_start.floor("D")
        weather_days = max(
            days + 1,
            ceil(max((requested_end - current_day_start) / pd.Timedelta(days=1), 0)),
        )
        weather = self._weather_for_model_grid(
            start=model_start,
            horizon=model_horizon,
            forecast_days=weather_days,
        )
        weather = self.bias_corrector.apply(weather)
        aligned_weather = self._align_weather_grid(
            weather,
            start=model_start,
            horizon=model_horizon,
        )
        aligned_weather = self._merge_known_future(aligned_weather, known_future)
        if self.selected_candidate in (ML_CANDIDATE, BLEND_CANDIDATE):
            features = self.feature_builder.build(aligned_weather)
            validate_timestamp_grid(
                features,
                interval_minutes=interval_minutes,
                expected_start=model_start,
                expected_length=model_horizon,
                label="forecast features",
            )
            forecast = self.backend.predict(features, horizon=model_horizon)
            if self.selected_candidate == BLEND_CANDIDATE:
                if self._blend_weight is None:
                    raise ForecastAlignmentError("blend weight is unavailable")
                forecast = self._sanitize_point_forecast(forecast)
                regression = self._regression_values(aligned_weather).to_numpy()
                forecast["yhat"] = (
                    self._blend_weight * forecast["yhat"].to_numpy()
                    + (1.0 - self._blend_weight) * regression
                )
        else:
            forecast = self._predict_benchmark(
                aligned_weather, start=model_start, horizon=model_horizon
            )
        validate_timestamp_grid(
            forecast,
            interval_minutes=interval_minutes,
            expected_start=model_start,
            expected_length=model_horizon,
            label="backend forecast",
        )
        forecast = self._sanitize_point_forecast(forecast)
        forecast = self.conformal.apply(forecast)
        forecast = forecast.iloc[
            skipped_steps : skipped_steps + returned_horizon
        ].reset_index(drop=True)
        validate_timestamp_grid(
            forecast,
            interval_minutes=interval_minutes,
            expected_start=requested_start,
            expected_length=returned_horizon,
            label="returned forecast",
        )
        if (
            pd.to_datetime(forecast["ds"].iloc[-1], utc=True) + interval
            != requested_end
        ):
            raise ForecastAlignmentError(
                "returned forecast has an invalid end boundary"
            )
        if self.config.forecast_type == MLForecastType.SOLAR:
            forecast = self._apply_solar_constraints(forecast)
            forecast = self._apply_pv_curtailment(
                forecast, dynamic_export_limits=dynamic_export_limits
            )
        return forecast

    def _history_covariates(self, history_df: pd.DataFrame) -> pd.DataFrame | None:
        """Return known covariates averaged onto the model grid, or ``None``.

        Raises:
            ValueError: If a configured covariate column is missing.
        """
        names = list(self.config.known_covariates)
        if not names:
            return None
        missing = [name for name in names if name not in history_df.columns]
        if missing:
            raise ValueError(f"history is missing known covariates {missing}")
        frame = history_df[["ds", *names]].copy()
        frame["ds"] = pd.to_datetime(frame["ds"], utc=True).dt.floor(
            f"{self.config.interval_minutes}min"
        )
        for name in names:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        return frame.groupby("ds", as_index=False)[names].mean()

    def _merge_known_future(
        self, weather: pd.DataFrame, known_future: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Add known covariates to the model-grid weather.

        Raises:
            ForecastAlignmentError: If a configured covariate is missing or
                not finite on any model-grid row.
        """
        names = list(self.config.known_covariates)
        if not names:
            return weather
        if known_future is None:
            raise ForecastAlignmentError(
                f"known_future is required for covariates {names}"
            )
        missing = [name for name in names if name not in known_future.columns]
        if missing or "ds" not in known_future.columns:
            raise ForecastAlignmentError(
                f"known_future is missing columns {missing or ['ds']}"
            )
        future = known_future[["ds", *names]].copy()
        future["ds"] = pd.to_datetime(future["ds"], utc=True)
        if future["ds"].duplicated().any():
            raise ForecastAlignmentError("known_future has duplicate timestamps")
        grid = pd.to_datetime(weather["ds"], utc=True)
        values = (
            future.set_index("ds")[names]
            .apply(pd.to_numeric, errors="coerce")
            .reindex(grid)
        )
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ForecastAlignmentError(
                "known_future does not cover every model-grid row with finite values"
            )
        output = weather.copy()
        for name in names:
            output[name] = values[name].to_numpy(dtype=float)
        return output

    @staticmethod
    def _sanitize_point_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
        """Enforce the non-negative physical contract for point forecasts."""
        if "yhat" not in forecast.columns:
            raise ForecastAlignmentError("backend forecast is missing 'yhat'")

        output = forecast.copy()
        numeric = pd.to_numeric(output["yhat"], errors="coerce")
        for position, value in enumerate(numeric):
            if pd.isna(value) or value in (float("inf"), float("-inf")):
                timestamp = (
                    output["ds"].iloc[position] if "ds" in output.columns else None
                )
                timestamp_text = (
                    pd.to_datetime(timestamp, utc=True).isoformat()
                    if timestamp is not None
                    else "unknown"
                )
                raw_value = output["yhat"].iloc[position]
                raise ForecastAlignmentError(
                    "backend forecast yhat is non-finite at "
                    f"index={position} timestamp={timestamp_text} value={raw_value!r}"
                )
        output["yhat"] = numeric.clip(lower=0.0)
        return output

    def _weather_for_model_grid(
        self,
        *,
        start: pd.Timestamp,
        horizon: int,
        forecast_days: int,
    ) -> pd.DataFrame:
        """Combine archive and forecast weather for the complete model grid.

        Recursive backends must advance through every interval after their
        training cutoff, even though only the requested future slice is
        returned. Historical weather fills any elapsed prefix no longer
        supplied by the forecast endpoint.
        """
        interval_minutes = self.config.interval_minutes
        interval = pd.Timedelta(minutes=interval_minutes)
        end = start + (horizon - 1) * interval
        current_day_start = pd.Timestamp.now(tz="UTC").floor("D")
        past_days = min(
            self.weather_client.config.recent_forecast_past_days,
            max(ceil((current_day_start - start) / pd.Timedelta(days=1)), 0),
        )
        forecast = self.weather_client.fetch_forecast(
            days=forecast_days,
            past_days=past_days,
        )
        forecast = self.weather_client.resample_weather(forecast, interval_minutes)
        if "ds" not in forecast.columns or forecast.empty:
            raise ForecastAlignmentError("weather forecast has no timestamps")
        forecast = forecast.copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"], utc=True)
        forecast_start = pd.to_datetime(forecast["ds"].min(), utc=True)

        frames: list[pd.DataFrame] = []
        historical_end = min(end, forecast_start - interval)
        if start <= historical_end:
            historical = self.weather_client.fetch_historical(
                start.date(), historical_end.date()
            )
            historical = self.weather_client.resample_weather(
                historical, interval_minutes
            )
            frames.append(historical)
        frames.append(forecast)

        combined = pd.concat(frames, ignore_index=True)
        if "ds" not in combined.columns:
            raise ForecastAlignmentError("weather data is missing 'ds'")
        combined["ds"] = pd.to_datetime(combined["ds"], utc=True)
        combined = (
            combined.drop_duplicates(subset="ds", keep="last")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        if len(frames) == 2:
            combined = self._fill_weather_seam(
                combined, seam_end=forecast_start, interval=interval
            )
        return combined

    @staticmethod
    def _fill_weather_seam(
        weather: pd.DataFrame, *, seam_end: pd.Timestamp, interval: pd.Timedelta
    ) -> pd.DataFrame:
        """Interpolate the short gap between archive and forecast weather.

        Only grid slots strictly between the last archive row and the first
        forecast row are added, and only when that seam is at most
        ``MAX_WEATHER_SEAM``. Gaps inside either source are left missing so
        grid alignment still rejects them.

        Args:
            weather: Combined, de-duplicated weather sorted by ``ds``.
            seam_end: First timestamp supplied by the forecast source.
            interval: Model grid interval.

        Returns:
            Weather with the seam slots interpolated in time, or unchanged.
        """
        before = weather.loc[weather["ds"] < seam_end, "ds"]
        if before.empty:
            return weather
        seam_start = before.max()
        if not interval < seam_end - seam_start <= MAX_WEATHER_SEAM:
            return weather
        missing = pd.date_range(
            seam_start + interval, seam_end - interval, freq=interval, tz="UTC"
        )
        indexed = weather.set_index("ds")
        numeric = indexed.select_dtypes("number").columns
        bridge = indexed.loc[[seam_start, seam_end], numeric]
        filled = (
            bridge.reindex(bridge.index.union(missing))
            .interpolate(method="time")
            .loc[missing]
        )
        filled.index.name = "ds"
        return (
            pd.concat([weather, filled.reset_index()], ignore_index=True)
            .sort_values("ds")
            .reset_index(drop=True)
        )

    def _align_weather_grid(
        self,
        weather: pd.DataFrame,
        *,
        start: pd.Timestamp,
        horizon: int,
    ) -> pd.DataFrame:
        """Select weather rows for the exact model forecast grid.

        Args:
            weather: Resampled weather forecast containing ``ds``.
            start: First post-training model timestamp.
            horizon: Number of model-grid rows required.

        Returns:
            Weather dataframe ordered on the exact contiguous model grid.

        Raises:
            ForecastAlignmentError: If weather coverage is missing or duplicated.
        """
        if "ds" not in weather.columns:
            raise ForecastAlignmentError("weather forecast is missing 'ds'")
        normalized = weather.copy()
        normalized["ds"] = pd.to_datetime(normalized["ds"], utc=True)
        if normalized["ds"].duplicated().any():
            raise ForecastAlignmentError(
                "weather forecast contains duplicate timestamps"
            )
        normalized = normalized.sort_values("ds").set_index("ds")
        expected = pd.date_range(
            start=start,
            periods=horizon,
            freq=f"{self.config.interval_minutes}min",
            tz="UTC",
        )
        available = pd.DatetimeIndex(normalized.index)
        missing = expected.difference(available)
        if not missing.empty:
            preview = ", ".join(timestamp.isoformat() for timestamp in missing[:3])
            raise ForecastAlignmentError(
                f"weather forecast is missing {len(missing)} required timestamps: "
                f"{preview}"
            )
        aligned = normalized.reindex(expected)
        aligned.index.name = "ds"
        return aligned.reset_index()

    def _restore_existing_manifest(self) -> None:
        """Load persisted manifest state when available."""
        manifest = self.storage.load_manifest()
        if manifest is None:
            return
        if manifest.metadata.get("timezone") != self.config.timezone:
            return
        if manifest.metadata.get("history_policy_version") != HISTORY_POLICY_VERSION:
            return
        if manifest.metadata.get("preserve_gaps") != self.config.preserve_gaps:
            return
        if (
            manifest.metadata.get("max_bridged_gap_intervals", 0)
            != self.config.max_bridged_gap_intervals
        ):
            return
        if not _selection_settings_match(manifest, self.config):
            return
        if not _structure_matches(manifest, self.config):
            return
        if not _covariates_match(manifest, self.config):
            return
        if manifest.contract_version != FORECAST_CONTRACT_VERSION:
            return
        if (
            manifest.metadata.get("sanitized_conformal_state_version")
            != SANITIZED_CONFORMAL_STATE_VERSION
        ):
            return
        self._restore_from_manifest(manifest)

    def _restore_from_manifest(self, manifest: MLModelManifest) -> None:
        """Restore backend, interval, and correction state from a manifest."""
        if manifest.contract_version != FORECAST_CONTRACT_VERSION:
            raise ForecastAlignmentError(
                "stored model uses forecast contract "
                f"{manifest.contract_version}; contract "
                f"{FORECAST_CONTRACT_VERSION} requires a full retrain"
            )
        if (
            manifest.metadata.get("sanitized_conformal_state_version")
            != SANITIZED_CONFORMAL_STATE_VERSION
        ):
            raise ForecastAlignmentError(
                "stored model conformal state predates point-forecast sanitation; "
                "a full retrain is required"
            )
        if manifest.metadata.get("timezone") != self.config.timezone:
            raise ForecastAlignmentError(
                "stored model timezone requires a full retrain"
            )
        if manifest.metadata.get("history_policy_version") != HISTORY_POLICY_VERSION:
            raise ForecastAlignmentError(
                "stored model history policy requires a full retrain"
            )
        if manifest.metadata.get("preserve_gaps") != self.config.preserve_gaps:
            raise ForecastAlignmentError(
                "stored model gap-preservation mode requires a full retrain"
            )
        if (
            manifest.metadata.get("max_bridged_gap_intervals", 0)
            != self.config.max_bridged_gap_intervals
        ):
            raise ForecastAlignmentError(
                "stored model gap-bridging limit requires a full retrain"
            )
        if not _selection_settings_match(manifest, self.config):
            raise ForecastAlignmentError(
                "stored model candidate-selection settings require a full retrain"
            )
        if not _structure_matches(manifest, self.config):
            raise ForecastAlignmentError(
                "stored model hybrid structure requires a full retrain"
            )
        if not _covariates_match(manifest, self.config):
            raise ForecastAlignmentError(
                "stored model known covariates require a full retrain"
            )
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
        if manifest.training_end is None:
            raise ForecastAlignmentError("stored model has no training cutoff")
        self._training_end = pd.to_datetime(manifest.training_end, utc=True)
        self.training_bridged_rows = int(manifest.metadata.get("bridged_rows", 0))
        self._restore_selection(manifest)
        self.conformal = SplitConformalCalibrator.from_dict(
            manifest.interval_levels, manifest.conformal_quantiles
        )
        weather_bias = manifest.metadata.get("weather_bias")
        if isinstance(weather_bias, dict):
            self.bias_corrector.load_dict(weather_bias)

    @property
    def training_end(self) -> datetime | None:
        """Return the UTC cutoff used to build the active model."""
        if self._training_end is None:
            return None
        return cast(datetime, self._training_end.to_pydatetime())

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

    def _new_regression(self) -> DegreeHourRegression:
        """Return an unfitted degree-hour regression for this configuration."""
        return DegreeHourRegression(
            base_temperature_c=self.config.regression_base_temperature_c,
            extra_features=tuple(self.config.regression_extra_features),
            timezone=self.config.timezone,
        )

    def _select_candidate(
        self,
        history: pd.DataFrame,
        features: pd.DataFrame,
        bridged: pd.Series,
        calibration_actual: pd.Series,
        ml_holdout: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Backtest the candidates over recent day-ahead origins and pick one.

        Every candidate forecasts the next ``selection_horizon_hours`` from
        local midnight of each of the last ``selection_backtest_days`` days,
        trained only on rows before that origin. Prediction intervals are
        calibrated on leave-one-origin-out residuals: at each origin, the
        candidate the rule picks from the *other* origins. That calibrates the
        selection procedure that is actually served, on residuals that did
        not choose it. With fewer than ``selection_min_origins`` usable
        origins the ML model is kept and the calibration tail is used.

        Args:
            history: Complete bridged history with weather (and covariates).
            features: Features built from ``history``, row for row.
            bridged: Rows of ``history`` that were interpolated, not measured.
            calibration_actual: Measured calibration-tail targets.
            ml_holdout: The ML model's calibration-tail forecast.

        Returns:
            Actual and predicted values to calibrate intervals on.
        """
        self.selected_candidate = ML_CANDIDATE
        self.candidate_metrics = {}
        self.selection_reason = None
        self._regression = None
        self._blend_weight = None
        default = (calibration_actual, ml_holdout)
        if not self.config.candidate_selection:
            return default
        backtest = self._backtest(history, features, bridged)
        minimum = self.config.selection_min_origins

        def covered(name: str) -> bool:
            return sum(name in window for window in backtest.values()) >= minimum

        names = [
            name
            for name in self.config.selection_candidates
            if name != BLEND_CANDIDATE and (name == ML_CANDIDATE or covered(name))
        ]
        # The blend needs regression forecasts even when the regression alone
        # is not a candidate.
        blend = BLEND_CANDIDATE in self.config.selection_candidates and covered(
            DEGREE_HOUR_CANDIDATE
        )
        required = names + ([DEGREE_HOUR_CANDIDATE] if blend else [])
        origins = [
            origin
            for origin, window in backtest.items()
            if all(name in window for name in required)
        ]
        if len(origins) < minimum:
            self.selection_reason = "backtest_too_short"
            return default
        windows = {origin: dict(backtest[origin]) for origin in origins}
        if blend:
            windows = self._with_blend(windows, origins, held_out=None)
            names.append(BLEND_CANDIDATE)

        metrics = self._score_backtest(windows, names, origins)
        winner = self._select(metrics)
        choices = self._leave_one_out_choices(windows, names, origins, winner, blend)
        conformal_actual = [windows[o]["actual"] for o in origins]
        conformal_predicted = [windows[o][choices[o]] for o in origins]

        self.selected_candidate = winner
        self.candidate_metrics = {name: m.as_dict() for name, m in metrics.items()}
        if not any(
            bias_eligible(
                m,
                tolerance=self.config.selection_bias_tolerance,
                floor=self._bias_floor(),
            )
            for m in metrics.values()
        ):
            self.selection_reason = "no_candidate_within_bias_guard"
        if winner in (DEGREE_HOUR_CANDIDATE, BLEND_CANDIDATE):
            # Refit on all measured rows after selection.
            self._regression = self._new_regression()
        if winner == BLEND_CANDIDATE:
            self._blend_weight = self._window_blend_weight(windows, origins)
        return (
            pd.Series(np.concatenate(conformal_actual)),
            pd.Series(np.concatenate(conformal_predicted)),
        )

    def _with_blend(
        self,
        windows: dict[pd.Timestamp, dict[str, np.ndarray]],
        origins: list[pd.Timestamp],
        *,
        held_out: pd.Timestamp | None,
    ) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
        """Add blend predictions, each weighted without its own origin.

        With ``held_out`` set, that origin's actuals are excluded from every
        weight too, so it cannot influence a choice made for it.
        """
        output = {origin: dict(window) for origin, window in windows.items()}
        for origin in origins:
            fit_on = [o for o in origins if o not in (origin, held_out)]
            weight = self._window_blend_weight(windows, fit_on)
            output[origin][BLEND_CANDIDATE] = (
                weight * windows[origin][ML_CANDIDATE]
                + (1.0 - weight) * windows[origin][DEGREE_HOUR_CANDIDATE]
            )
        return output

    def _leave_one_out_choices(
        self,
        windows: dict[pd.Timestamp, dict[str, np.ndarray]],
        names: list[str],
        origins: list[pd.Timestamp],
        winner: str,
        blend: bool = False,
    ) -> dict[pd.Timestamp, str]:
        """Return, per origin, the candidate chosen from the other origins.

        Blend predictions are rebuilt per held-out origin so its actuals take
        no part in the weights. With a single origin there is nothing to leave
        out; it keeps ``winner``.
        """
        choices: dict[pd.Timestamp, str] = {}
        for origin in origins:
            others = [o for o in origins if o != origin]
            if not others:
                choices[origin] = winner
                continue
            fold = (
                self._with_blend(windows, origins, held_out=origin)
                if blend
                else windows
            )
            choices[origin] = self._select(self._score_backtest(fold, names, others))
        return choices

    def _bias_floor(self) -> float:
        """Return the bias floor in target units (kWh) per interval."""
        return self.config.selection_bias_floor_kw * self.config.interval_minutes / 60

    def _select(self, metrics: dict[str, CandidateMetrics]) -> str:
        return select_candidate(
            metrics,
            bias_tolerance=self.config.selection_bias_tolerance,
            bias_floor=self._bias_floor(),
        )

    def _score_backtest(
        self,
        windows: dict[pd.Timestamp, dict[str, np.ndarray]],
        names: list[str],
        origins: list[pd.Timestamp],
    ) -> dict[str, CandidateMetrics]:
        steps = max(1, 60 // self.config.interval_minutes)
        scored: dict[str, CandidateMetrics] = {}
        for name in names:
            metrics = score_windows(
                [(windows[o]["actual"], windows[o][name]) for o in origins],
                window_steps=steps,
            )
            if metrics is not None:
                scored[name] = metrics
        return scored

    @staticmethod
    def _window_blend_weight(
        windows: dict[pd.Timestamp, dict[str, np.ndarray]],
        origins: list[pd.Timestamp],
    ) -> float:
        if not origins:
            return 1.0
        return fit_blend_weight(
            [windows[o]["actual"] for o in origins],
            [windows[o][ML_CANDIDATE] for o in origins],
            [windows[o][DEGREE_HOUR_CANDIDATE] for o in origins],
        )

    def _backtest_origins(self, last_measured: pd.Timestamp) -> list[pd.Timestamp]:
        """Local midnights whose full horizon ends at or before the data end."""
        interval = pd.Timedelta(minutes=self.config.interval_minutes)
        horizon = pd.Timedelta(hours=self.config.selection_horizon_hours)
        latest_start = (last_measured + interval - horizon).tz_convert(
            self.config.timezone
        )
        latest = latest_start.normalize()
        return sorted(
            (latest - pd.DateOffset(days=offset)).tz_convert("UTC")
            for offset in range(self.config.selection_backtest_days)
        )

    def _backtest(
        self,
        history: pd.DataFrame,
        features: pd.DataFrame,
        bridged: pd.Series,
    ) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
        """Forecast each backtest origin with every enabled candidate.

        Origins whose window is not on the grid, has under 90 % measured rows,
        or cannot be forecast by the ML model are left out. Benchmarks that
        cannot forecast an origin are missing from that origin's entry.

        Returns:
            Per origin: ``actual`` (NaN where not measured) and one array per
            candidate that produced a forecast.
        """
        frame = history.reset_index(drop=True).copy()
        frame["ds"] = pd.to_datetime(frame["ds"], utc=True)
        feature_rows = features.reset_index(drop=True)
        bridged_rows = bridged.reset_index(drop=True).astype(bool)
        target = pd.to_numeric(frame["y"], errors="coerce")
        measured = target.where(~bridged_rows)
        if not measured.notna().any():
            return {}
        interval = pd.Timedelta(minutes=self.config.interval_minutes)
        steps = self.config.selection_horizon_hours * 60 // self.config.interval_minutes
        position = pd.Series(frame.index, index=frame["ds"])
        last_measured = frame.loc[measured.notna(), "ds"].max()
        wanted = set(self.config.selection_candidates)
        results: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
        for origin in self._backtest_origins(last_measured):
            grid = pd.date_range(origin, periods=steps, freq=interval, tz="UTC")
            rows = position.reindex(grid)
            if rows.isna().any():
                continue
            index = rows.astype(int).to_numpy()
            window = frame.iloc[index].reset_index(drop=True)
            actual = measured.iloc[index].to_numpy(dtype=float)
            if np.isfinite(actual).mean() < 0.9:
                continue
            train_mask = (frame["ds"] < origin) & target.notna()
            train = frame.loc[train_mask].reset_index(drop=True)
            if train.empty or train["ds"].iloc[-1] != origin - interval:
                continue
            ml = self._backtest_ml(
                train,
                feature_rows.loc[train_mask].reset_index(drop=True),
                feature_rows.iloc[index].reset_index(drop=True),
                steps,
            )
            if ml is None:
                continue
            entry: dict[str, np.ndarray] = {"actual": actual, ML_CANDIDATE: ml}
            train_measured = frame.loc[train_mask & ~bridged_rows].reset_index(
                drop=True
            )
            if wanted & {DEGREE_HOUR_CANDIDATE, BLEND_CANDIDATE}:
                regression = self._new_regression()
                if regression.fit(train_measured):
                    values = regression.predict(window).to_numpy(dtype=float)
                    if np.isfinite(values).all():
                        entry[DEGREE_HOUR_CANDIDATE] = values
            if BASELINE_NAME in wanted:
                try:
                    baseline = local_slot_weekday_class_median(
                        train_measured[["ds", "y"]],
                        origin=origin.to_pydatetime(),
                        periods=steps,
                        interval_minutes=self.config.interval_minutes,
                        timezone=self.config.timezone,
                    )
                except ForecastAlignmentError:
                    pass
                else:
                    entry[BASELINE_NAME] = baseline["yhat"].to_numpy(dtype=float)
            results[origin] = entry
        return results

    def _backtest_ml(
        self,
        train: pd.DataFrame,
        train_features: pd.DataFrame,
        window_features: pd.DataFrame,
        steps: int,
    ) -> np.ndarray | None:
        """Fit a fresh backend on ``train`` and forecast the next ``steps`` rows.

        Returns:
            Non-negative predictions, or ``None`` when the backend cannot be
            trained this early or produces non-finite values.
        """
        backend = create_backend(self.config)
        try:
            backend.fit(train, train_features, train.iloc[0:0])
            forecast = backend.predict(window_features, horizon=steps)
            forecast = self._sanitize_point_forecast(forecast)
        except (ValueError, ForecastAlignmentError):
            return None
        values = forecast["yhat"].to_numpy(dtype=float)
        return values if len(values) == steps else None

    def _regression_values(self, weather: pd.DataFrame) -> pd.Series:
        """Predict the fitted degree-hour regression on the model grid."""
        if self._regression is None:
            raise ForecastAlignmentError("degree-hour regression is unavailable")
        values = self._regression.predict(weather.reset_index(drop=True))
        if values.isna().any():
            raise ForecastAlignmentError(
                "degree-hour regression is missing outdoor temperature"
            )
        return values

    def _predict_benchmark(
        self, weather: pd.DataFrame, *, start: pd.Timestamp, horizon: int
    ) -> pd.DataFrame:
        """Forecast the model grid with the selected benchmark.

        Args:
            weather: Bias-corrected weather aligned to the model grid.
            start: First post-training model timestamp.
            horizon: Number of model-grid rows.

        Returns:
            Dataframe with ``ds`` and ``yhat`` on the model grid.

        Raises:
            ForecastAlignmentError: If the benchmark cannot cover the grid.
        """
        grid = pd.date_range(
            start=start,
            periods=horizon,
            freq=f"{self.config.interval_minutes}min",
            tz="UTC",
        )
        if self.selected_candidate == DEGREE_HOUR_CANDIDATE:
            values = self._regression_values(weather)
        elif self.selected_candidate == BASELINE_NAME:
            history = self.storage.load_training_frame()
            if history is None:
                raise ForecastAlignmentError("baseline training history is unavailable")
            values = local_slot_weekday_class_median(
                history,
                origin=start.to_pydatetime(),
                periods=horizon,
                interval_minutes=self.config.interval_minutes,
                timezone=self.config.timezone,
            )["yhat"]
        else:
            raise ForecastAlignmentError(
                f"unknown selected candidate {self.selected_candidate!r}"
            )
        return pd.DataFrame({"ds": grid, "yhat": values.to_numpy(dtype=float)})

    def _restore_selection(self, manifest: MLModelManifest) -> None:
        """Restore the selected candidate and its state from a manifest."""
        selection = manifest.metadata.get("candidate_selection")
        selection = selection if isinstance(selection, dict) else {}
        self.selected_candidate = str(selection.get("selected") or ML_CANDIDATE)
        candidates = selection.get("candidates")
        self.candidate_metrics = candidates if isinstance(candidates, dict) else {}
        reason = selection.get("reason")
        self.selection_reason = reason if isinstance(reason, str) else None
        self._regression = None
        self._blend_weight = None
        if self.selected_candidate not in SELECTION_CANDIDATES:
            raise ForecastAlignmentError(
                f"stored candidate {self.selected_candidate!r} is unknown; a full "
                "retrain is required"
            )
        if self.selected_candidate in (DEGREE_HOUR_CANDIDATE, BLEND_CANDIDATE):
            payload = manifest.metadata.get("degree_hour_regression")
            if not isinstance(payload, dict):
                raise ForecastAlignmentError(
                    "stored degree-hour regression is missing; a full retrain is "
                    "required"
                )
            self._regression = DegreeHourRegression.from_dict(payload)
        if self.selected_candidate == BLEND_CANDIDATE:
            weight = selection.get("blend_weight")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, int | float)
                or not 0.0 <= float(weight) <= 1.0
            ):
                raise ForecastAlignmentError(
                    "stored blend weight is invalid; a full retrain is required"
                )
            self._blend_weight = float(weight)

    def predict_baseline(
        self,
        *,
        days: int,
        origin: datetime,
        timezone: str | None = None,
    ) -> pd.DataFrame:
        """Generate the leakage-safe fallback forecast from persisted history."""
        history = self.storage.load_training_frame()
        if history is None:
            raise ForecastAlignmentError("baseline training history is unavailable")
        periods = days * (24 * 60 // self.config.interval_minutes)
        return local_slot_weekday_class_median(
            history,
            origin=origin,
            periods=periods,
            interval_minutes=self.config.interval_minutes,
            timezone=timezone or self.config.timezone,
        )

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
        return prepared.reset_index(drop=True)

    def _gap_safe_training_split(
        self,
        prepared: pd.DataFrame,
        features: pd.DataFrame,
        bridged: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Keep all observations while reserving a contiguous calibration tail.

        When ``bridged`` marks interpolated rows, the calibration boundary is
        moved forward until both the first calibration row and the row before
        it are measured, so no bridge interpolates across the split and
        calibration targets cannot leak into training rows.
        """
        valid = pd.to_numeric(prepared["y"], errors="coerce").notna()
        if not valid.any():
            raise ValueError("training history has no usable target observations")
        observed = prepared.loc[valid].reset_index(drop=True)
        observed_features = features.loc[valid].reset_index(drop=True)
        if len(observed) < 4:
            raise ValueError("at least four rows are required for ML training")

        groups = (~valid).cumsum()
        latest_group = groups.loc[valid].iloc[-1]
        latest_indices = prepared.index[valid & groups.eq(latest_group)]
        minimum_training_rows = max(
            1,
            int(getattr(self.backend, "minimum_contiguous_training_rows", 1)),
        )
        if len(latest_indices) <= minimum_training_rows:
            raise ValueError(
                "latest contiguous history requires at least "
                f"{minimum_training_rows + 1} rows to reserve calibration data"
            )
        calibration_size = max(
            1, int(len(latest_indices) * self.config.calibration_fraction)
        )
        calibration_size = min(
            calibration_size, len(latest_indices) - minimum_training_rows
        )
        calibration_indices = latest_indices[-calibration_size:]
        if bridged is not None:
            first = 0
            while first < len(calibration_indices) and (
                bool(bridged.loc[calibration_indices[first]])
                or bool(bridged.get(calibration_indices[first] - 1, False))
            ):
                first += 1
            if first == len(calibration_indices):
                raise ValueError(
                    "calibration tail has no measured boundary outside bridged gaps"
                )
            calibration_indices = calibration_indices[first:]
        calibration_start = calibration_indices[0]
        train_mask = valid & (prepared.index < calibration_start)
        return (
            observed,
            observed_features,
            prepared.loc[train_mask].reset_index(drop=True),
            prepared.loc[calibration_indices].reset_index(drop=True),
        )

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


def _selection_settings(config: KPowerMLConfig) -> dict[str, Any]:
    """Return the settings that define which candidates are trained."""
    settings: dict[str, Any] = {"enabled": config.candidate_selection}
    if config.candidate_selection:
        settings["regression_base_temperature_c"] = config.regression_base_temperature_c
        settings["regression_extra_features"] = list(config.regression_extra_features)
        settings["enabled_candidates"] = list(config.selection_candidates)
        settings["backtest_days"] = config.selection_backtest_days
        settings["horizon_hours"] = config.selection_horizon_hours
        settings["min_origins"] = config.selection_min_origins
        settings["bias_tolerance"] = config.selection_bias_tolerance
        settings["bias_floor_kw"] = config.selection_bias_floor_kw
    return settings


def _selection_settings_match(
    manifest: MLModelManifest, config: KPowerMLConfig
) -> bool:
    """Return whether a stored artifact was trained with this selection setup."""
    stored = manifest.metadata.get("candidate_selection")
    stored = stored if isinstance(stored, dict) else {}
    expected = _selection_settings(config)
    return all(
        stored.get(key, False if key == "enabled" else None) == value
        for key, value in expected.items()
    )


def _structure_matches(manifest: MLModelManifest, config: KPowerMLConfig) -> bool:
    """Return whether a stored artifact was trained with this hybrid structure.

    Artifacts written before the setting existed used the recursive structure.
    """
    stored = manifest.metadata.get(
        "hybrid_structure", HybridStructure.RECURSIVE_SEASONAL_NAIVE.value
    )
    if stored != config.hybrid_structure.value:
        return False
    if config.hybrid_structure == HybridStructure.PROFILE_DIRECT:
        return (
            manifest.metadata.get("profile_lookback_days")
            == config.profile_lookback_days
            and manifest.metadata.get("profile_class_prior_days")
            == config.profile_class_prior_days
            and manifest.metadata.get("profile_smoothing_minutes")
            == config.profile_smoothing_minutes
        )
    return True


def _covariates_match(manifest: MLModelManifest, config: KPowerMLConfig) -> bool:
    """Return whether a stored artifact was trained on these known covariates."""
    return bool(
        manifest.metadata.get("known_covariates", []) == list(config.known_covariates)
    )
