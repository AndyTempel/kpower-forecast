"""Stable first-order thermal response model for sparse real observations.

The effective HVAC gain is a temperature response per electric watt. It includes
unknown heat-pump efficiency and distribution effects and is not physical COP.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kpower_forecast import __version__
from kpower_forecast.weather_client import WeatherClient, WeatherConfig

from .fitting import (
    ChainSegment,
    FitCandidate,
    build_chains,
    build_windows,
    fit_candidate,
    fit_output_error,
)
from .weather import (
    exact_future_outdoor_grid,
    fetch_future_weather,
    fetch_historical_weather,
    mean_outdoor_temperature,
)

# 2: multi-step output-error fit, quality labels and regularised fits.
THERMAL_CONTRACT_VERSION = 2

ThermalFitQuality = Literal["accepted", "low_identifiability"]


def _utc(value: datetime) -> datetime:
    """Require an aware instant and normalize it to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("thermal timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


class ThermalModelConfig(BaseModel):
    """Named conservative identification and prediction limits.

    Variation gates avoid calling a constant-temperature, constant-drive week
    an identified building. Bounds allow ordinary houses while rejecting very
    rapid or nearly unobservable response in the initial heating regime.
    """

    model_config = ConfigDict(frozen=True)

    min_transition_minutes: float = Field(default=5.0, gt=0)
    max_transition_hours: float = Field(default=6.0, gt=0)
    min_hvac_coverage_ratio: float = Field(default=0.90, ge=0, le=1)
    min_transitions: int = Field(default=40, ge=3)
    min_span_hours: float = Field(default=72.0, gt=0)
    min_indoor_range_c: float = Field(default=0.5, gt=0)
    min_outdoor_range_c: float = Field(default=2.0, gt=0)
    min_delta_std_c: float = Field(default=0.5, gt=0)
    hvac_on_threshold_w: float = Field(default=100.0, ge=0)
    min_on_transitions: int = Field(default=8, ge=1)
    min_off_transitions: int = Field(default=8, ge=1)
    min_hvac_power_std_w: float = Field(default=150.0, gt=0)
    min_time_constant_hours: float = Field(default=2.0, gt=0)
    max_time_constant_hours: float = Field(default=240.0, gt=0)
    max_equilibrium_offset_c: float = Field(default=12.0, gt=0)
    max_effective_gain_c_per_kw: float = Field(default=20.0, gt=0)
    holdout_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    min_holdout_transitions: int = Field(default=8, ge=2)
    max_holdout_bias_c: float = Field(default=1.0, gt=0)
    min_persistence_improvement_ratio: float = Field(default=0.05, ge=0, lt=1)
    interval_coverage: float = Field(default=0.9, gt=0, lt=1)
    interval_minutes: int = Field(default=15, ge=1)
    max_prediction_periods: int = Field(default=5 * 24 * 4, ge=1)
    fit_grid_points: int = Field(default=80, ge=10, le=1000)
    # Multi-step (output-error) training windows. Inputs across a short gap
    # are filled from the adjacent intervals; indoor targets never are.
    min_window_hours: float = Field(default=6.0, gt=0)
    max_window_hours: float = Field(default=24.0, gt=0)
    window_stride_hours: float = Field(default=3.0, gt=0)
    max_bridge_minutes: float = Field(default=30.0, ge=0)
    max_bridge_fraction: float = Field(default=0.1, ge=0, lt=1)
    # Weak log-normal prior: sigma ln(3) keeps 20-180 h inside one sigma.
    tau_prior_median_hours: float = Field(default=60.0, gt=0)
    tau_prior_log_sigma: float = Field(default=1.1, gt=0)
    min_effective_gain_c_per_kw: float = Field(default=0.05, gt=0)
    min_profile_relative_range: float = Field(default=0.01, ge=0, lt=1)
    # Acceptance is multi-step skill against persistence, not one-step error,
    # which quantised sensors make indistinguishable from persistence.
    skill_horizons_hours: tuple[int, ...] = (3, 6, 12)
    min_skill_horizons: int = Field(default=2, ge=1)
    min_horizon_origins: int = Field(default=4, ge=1)
    # A regularised low-identifiability fit needs far less history than an
    # accepted one; it is published with an explicit quality label.
    min_regularised_span_hours: float = Field(default=24.0, gt=0)
    min_regularised_transitions: int = Field(default=12, ge=3)
    low_identifiability_interval_scale: float = Field(default=1.5, ge=1)

    @model_validator(mode="after")
    def validate_ordering(self) -> ThermalModelConfig:
        """Reject reversed duration and time-constant ranges."""
        if self.min_transition_minutes >= 60 * self.max_transition_hours:
            raise ValueError("minimum transition duration must be below maximum")
        if self.min_time_constant_hours >= self.max_time_constant_hours:
            raise ValueError("minimum time constant must be below maximum")
        if self.min_window_hours >= self.max_window_hours:
            raise ValueError("minimum window must be below maximum")
        if not (
            self.min_time_constant_hours
            <= self.tau_prior_median_hours
            <= self.max_time_constant_hours
        ):
            raise ValueError("time-constant prior must lie inside the hard limits")
        if self.min_effective_gain_c_per_kw >= self.max_effective_gain_c_per_kw:
            raise ValueError("minimum gain must be below maximum")
        if (
            not self.skill_horizons_hours
            or any(hour <= 0 for hour in self.skill_horizons_hours)
            or len(set(self.skill_horizons_hours)) != len(self.skill_horizons_hours)
        ):
            raise ValueError("skill horizons must be positive and distinct")
        if self.min_skill_horizons > len(self.skill_horizons_hours):
            raise ValueError("more skill horizons required than evaluated")
        if self.min_regularised_span_hours > self.min_span_hours:
            raise ValueError("regularised span must not exceed accepted span")
        return self


class ThermalObservedTransition(BaseModel):
    """One real indoor transition with coverage-gated electrical exposure."""

    model_config = ConfigDict(frozen=True)

    start_at: datetime
    end_at: datetime
    indoor_start_c: float = Field(allow_inf_nan=False)
    indoor_end_c: float = Field(allow_inf_nan=False)
    hvac_electric_mean_w: float = Field(ge=0, allow_inf_nan=False)
    hvac_coverage_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_times(self) -> ThermalObservedTransition:
        """Require positive elapsed UTC time without changing source timestamps."""
        if _utc(self.end_at) <= _utc(self.start_at):
            raise ValueError("thermal transition must have positive elapsed time")
        return self

    @property
    def elapsed_hours(self) -> float:
        """Return exact source-observation separation in hours."""
        return (_utc(self.end_at) - _utc(self.start_at)).total_seconds() / 3600.0


class ThermalTrainingTransition(ThermalObservedTransition):
    """Observed transition enriched with package-owned outdoor weather."""

    outdoor_mean_c: float = Field(allow_inf_nan=False)


class ThermalModelDiagnostics(BaseModel):
    """Identification quality and chronological holdout evidence."""

    fitted: bool = False
    reliable_on_holdout: bool = False
    unreliable_reason: str | None = "insufficient_history"
    transition_count: int = 0
    rejected_short: int = 0
    rejected_long: int = 0
    rejected_hvac_coverage: int = 0
    rejected_overlap: int = 0
    training_span_hours: float = 0.0
    indoor_range_c: float = 0.0
    outdoor_range_c: float = 0.0
    delta_std_c: float = 0.0
    hvac_power_std_w: float = 0.0
    hvac_on_transitions: int = 0
    hvac_off_transitions: int = 0
    time_constant_hours: float | None = None
    effective_gain_c_per_kw: float | None = None
    equilibrium_offset_c: float | None = None
    holdout_count: int = 0
    holdout_mae_c: float | None = None
    holdout_bias_c: float | None = None
    persistence_mae_c: float | None = None
    residual_p90_c: float | None = None
    holdout_horizon_metrics: dict[str, dict[str, float | int]] = Field(
        default_factory=dict
    )
    # Published fit quality; None when no fit may be published.
    quality: ThermalFitQuality | None = None
    identifiability_flags: list[str] = Field(default_factory=list)
    fit_method: str = "output_error"
    window_count: int = 0
    window_reading_count: int = 0
    bridged_hours: float = 0.0
    skill_horizons_passed: int = 0
    holdout_start_at: datetime | None = None
    # The previous one-step transition fit, kept for comparison only.
    transition_fit: dict[str, float] | None = None


class ThermalPredictionInterval(BaseModel):
    """One future indoor-temperature state and empirical uncertainty band."""

    timestamp: datetime
    indoor_temperature_c: float = Field(allow_inf_nan=False)
    lower_temperature_c: float = Field(allow_inf_nan=False)
    upper_temperature_c: float = Field(allow_inf_nan=False)


class ThermalPrediction(BaseModel):
    """Passive trajectory under a caller-supplied future electric drive."""

    origin: datetime
    model_contract_version: int = THERMAL_CONTRACT_VERSION
    operating_regime: Literal["heating"] = "heating"
    outdoor_source: str
    hvac_drive_source: str
    intervals: list[ThermalPredictionInterval]


class KPowerThermalForecast:
    """Identify and persist a stable effective first-order heating response."""

    def __init__(
        self,
        *,
        model_id: str,
        authority_fingerprint: str,
        storage_path: Path | str,
        config: ThermalModelConfig | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        weather_config: WeatherConfig | None = None,
        weather_client: WeatherClient | None = None,
    ) -> None:
        if not model_id or not authority_fingerprint:
            raise ValueError("model ID and authority fingerprint are required")
        self.model_id = model_id
        self.authority_fingerprint = authority_fingerprint
        self.storage_path = Path(storage_path)
        self.config = config or ThermalModelConfig()
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        if weather_client is not None:
            client_latitude = float(weather_client.lat)
            client_longitude = float(weather_client.lon)
            if (
                latitude is not None
                and longitude is not None
                and (latitude != client_latitude or longitude != client_longitude)
            ):
                raise ValueError("injected weather client location differs from site")
            latitude, longitude = client_latitude, client_longitude
        if (
            latitude is not None
            and longitude is not None
            and not (
                math.isfinite(latitude)
                and math.isfinite(longitude)
                and -90 <= latitude <= 90
                and -180 <= longitude <= 180
            )
        ):
            raise ValueError("site weather coordinates are invalid")
        self.latitude = latitude
        self.longitude = longitude
        self.weather_client = weather_client
        if (
            self.weather_client is None
            and latitude is not None
            and longitude is not None
        ):
            resolved_weather_config = weather_config or WeatherConfig(
                cache_dir=self.storage_path / "weather_cache"
            )
            self.weather_client = WeatherClient(
                latitude, longitude, resolved_weather_config
            )
        self.diagnostics = ThermalModelDiagnostics()
        self._parameters: tuple[float, float, float] | None = None
        # Set by load() when a compatibility input changed since the save.
        self.needs_retrain = False

    def _eligible(
        self, transitions: list[ThermalObservedTransition]
    ) -> tuple[list[ThermalObservedTransition], list[ThermalObservedTransition]]:
        """Return chronological input and the rows the fit could use."""
        ordered = sorted(transitions, key=lambda item: _utc(item.start_at))
        eligible = [
            item
            for item in ordered
            if self.config.min_transition_minutes
            <= item.elapsed_hours * 60
            <= self.config.max_transition_hours * 60
            and item.hvac_coverage_ratio >= self.config.min_hvac_coverage_ratio
        ]
        return ordered, eligible

    def _with_weather(
        self, eligible: list[ThermalObservedTransition]
    ) -> list[ThermalTrainingTransition]:
        """Attach package-owned historical outdoor means to real transitions."""
        if self.weather_client is None:
            raise ValueError("site weather location is required")
        if not eligible:
            return []
        weather = fetch_historical_weather(
            self.weather_client,
            start_at=eligible[0].start_at,
            end_at=max(item.end_at for item in eligible),
        )
        return [
            ThermalTrainingTransition(
                **item.model_dump(),
                outdoor_mean_c=mean_outdoor_temperature(
                    weather, start_at=item.start_at, end_at=item.end_at
                ),
            )
            for item in eligible
        ]

    def _count_pre_rejections(
        self,
        diagnostics: ThermalModelDiagnostics,
        ordered: list[ThermalObservedTransition],
    ) -> None:
        """Add rows filtered before weather enrichment to the reject counters."""
        diagnostics.rejected_short += sum(
            item.elapsed_hours * 60 < self.config.min_transition_minutes
            for item in ordered
        )
        diagnostics.rejected_long += sum(
            item.elapsed_hours > self.config.max_transition_hours for item in ordered
        )
        diagnostics.rejected_hvac_coverage += sum(
            item.hvac_coverage_ratio < self.config.min_hvac_coverage_ratio
            for item in ordered
        )

    def train_with_weather(
        self, transitions: list[ThermalObservedTransition]
    ) -> ThermalModelDiagnostics:
        """Fetch historical site weather and fit without regularizing indoor targets.

        Args:
            transitions: Consecutive real observations with actual HVAC exposure.

        Returns:
            Model and chronological holdout diagnostics.

        Raises:
            ValueError: When weather location or required archive coverage is absent.
        """
        if self.weather_client is None:
            raise ValueError("site weather location is required")
        if not transitions:
            return self.train([])
        ordered, eligible = self._eligible(transitions)
        diagnostics = self.train(self._with_weather(eligible))
        self._count_pre_rejections(diagnostics, ordered)
        return diagnostics

    def evaluate_with_weather(
        self,
        transitions: list[ThermalObservedTransition],
        *,
        equilibrium_shift_c: float = 0.0,
    ) -> dict[str, dict[str, float | int]]:
        """Score the active fit on other real transitions at fixed horizons.

        Used to compare a zone's own fit with the aggregate fit shifted by the
        zone's offset on the same held-out readings.

        Args:
            transitions: Real transitions to replay, for example a zone holdout.
            equilibrium_shift_c: Constant offset added to the fitted equilibrium.

        Returns:
            Horizon metrics in the same shape as ``holdout_horizon_metrics``.
        """
        if self._parameters is None:
            raise ValueError("thermal model is not fitted")
        _, eligible = self._eligible(transitions)
        return self.evaluate_horizons(
            self._with_weather(eligible), equilibrium_shift_c=equilibrium_shift_c
        )

    def predict_with_weather(
        self,
        *,
        origin: datetime,
        initial_temperature_c: float,
        hvac_electric_power_w: list[float],
        hvac_drive_source: str,
        equilibrium_shift_c: float = 0.0,
    ) -> ThermalPrediction:
        """Fetch site forecast weather and simulate only its exact covered grid.

        Args:
            origin: Start of the first future electric-drive interval.
            initial_temperature_c: Fresh real indoor state supplied by caller.
            hvac_electric_power_w: Explicit future site HVAC electric drive.
            hvac_drive_source: Electrical forecast provenance.
            equilibrium_shift_c: Constant offset added to the fitted equilibrium,
                used to derive a zone trajectory from the aggregate fit.

        Returns:
            Passive thermal trajectory with outdoor model provenance.
        """
        if self.weather_client is None:
            raise ValueError("site weather location is required")
        forecast = fetch_future_weather(
            self.weather_client,
            periods=len(hvac_electric_power_w),
            interval_minutes=self.config.interval_minutes,
        )
        outdoor = exact_future_outdoor_grid(
            forecast,
            origin=origin,
            periods=len(hvac_electric_power_w),
            interval_minutes=self.config.interval_minutes,
        )
        return self.predict(
            origin=origin,
            initial_temperature_c=initial_temperature_c,
            outdoor_temperature_c=outdoor,
            hvac_electric_power_w=hvac_electric_power_w,
            outdoor_source=self.weather_client.effective_forecast_model_id(),
            hvac_drive_source=hvac_drive_source,
            equilibrium_shift_c=equilibrium_shift_c,
        )

    def _keep_previous(self, diagnostics: ThermalModelDiagnostics) -> None:
        """Expose a failed attempt only when no fit is active."""
        if self._parameters is None:
            self.diagnostics = diagnostics

    def train(
        self, transitions: list[ThermalTrainingTransition]
    ) -> ThermalModelDiagnostics:
        """Fit on chronological real-observation transitions with a blocked holdout.

        The primary fit simulates 6-24 h windows from real readings and scores
        every real reading in them (output error). Acceptance requires
        multi-step skill over persistence. A fit that is weakly identified or
        short of the accepted-history gates is still published as
        ``low_identifiability`` when it is not worse than persistence.

        Args:
            transitions: Sparse actual state transitions with outdoor and
                coverage-gated electric exposure supplied by the data adapter.

        Returns:
            Attempt diagnostics. A failed retry preserves any active fitted model.
        """
        ordered = sorted(transitions, key=lambda item: _utc(item.start_at))
        good: list[ThermalTrainingTransition] = []
        short = long = uncovered = overlap = 0
        previous_end: datetime | None = None
        for item in ordered:
            if previous_end is not None and _utc(item.start_at) < previous_end:
                overlap += 1
            elif item.elapsed_hours * 60 < self.config.min_transition_minutes:
                short += 1
            elif item.elapsed_hours > self.config.max_transition_hours:
                long += 1
            elif item.hvac_coverage_ratio < self.config.min_hvac_coverage_ratio:
                uncovered += 1
            else:
                good.append(item)
                previous_end = _utc(item.end_at)
        d = ThermalModelDiagnostics(
            transition_count=len(good),
            rejected_short=short,
            rejected_long=long,
            rejected_hvac_coverage=uncovered,
            rejected_overlap=overlap,
        )
        if not good:
            self._keep_previous(d)
            return d
        starts = np.array([item.indoor_start_c for item in good])
        ends = np.array([item.indoor_end_c for item in good])
        outdoors = np.array([item.outdoor_mean_c for item in good])
        power_kw = np.array([item.hvac_electric_mean_w / 1000 for item in good])
        hours = np.array([item.elapsed_hours for item in good])
        indoor = np.concatenate((starts, ends))
        d.training_span_hours = (
            _utc(good[-1].end_at) - _utc(good[0].start_at)
        ).total_seconds() / 3600
        d.indoor_range_c = float(np.ptp(indoor))
        d.outdoor_range_c = float(np.ptp(outdoors))
        d.delta_std_c = float(np.std(outdoors - starts))
        d.hvac_power_std_w = float(np.std(power_kw * 1000))
        d.hvac_on_transitions = int(
            np.count_nonzero(power_kw * 1000 >= self.config.hvac_on_threshold_w)
        )
        d.hvac_off_transitions = len(good) - d.hvac_on_transitions
        gate_reason: str | None
        if (
            len(good) < self.config.min_transitions
            or d.training_span_hours < self.config.min_span_hours
        ):
            gate_reason = "insufficient_history"
        elif d.indoor_range_c < self.config.min_indoor_range_c:
            gate_reason = "insufficient_temperature_variation"
        elif (
            d.outdoor_range_c < self.config.min_outdoor_range_c
            or d.delta_std_c < self.config.min_delta_std_c
            or d.hvac_on_transitions < self.config.min_on_transitions
            or d.hvac_off_transitions < self.config.min_off_transitions
            or d.hvac_power_std_w < self.config.min_hvac_power_std_w
        ):
            gate_reason = "insufficient_excitation"
        else:
            gate_reason = None
        d.unreliable_reason = gate_reason
        if (
            len(good) < self.config.min_regularised_transitions
            or d.training_span_hours < self.config.min_regularised_span_hours
        ):
            d.unreliable_reason = "insufficient_history"
            self._keep_previous(d)
            return d

        holdout_count = max(
            self.config.min_holdout_transitions,
            math.ceil(len(good) * self.config.holdout_fraction),
        )
        # A cold-start lineage may be too short to hold anything out. It can
        # then only become a low-identifiability fit, never an accepted one.
        has_holdout = (
            len(good) - holdout_count >= self.config.min_regularised_transitions
        )
        train_end = len(good) - holdout_count if has_holdout else len(good)
        transition_fit = fit_candidate(
            starts[:train_end],
            ends[:train_end],
            outdoors[:train_end],
            power_kw[:train_end],
            hours[:train_end],
            min_tau=self.config.min_time_constant_hours,
            max_tau=self.config.max_time_constant_hours,
            grid_points=self.config.fit_grid_points,
        )
        if transition_fit is not None:
            d.transition_fit = {
                "time_constant_hours": transition_fit.tau_hours,
                "effective_gain_c_per_kw": transition_fit.gain_c_per_kw,
                "equilibrium_offset_c": transition_fit.offset_c,
                "loss_c2": transition_fit.loss_c2,
            }
        primary = self._primary_fit(good[:train_end])
        if primary is None:
            # No window long enough to simulate, or no feasible coefficients.
            d.unreliable_reason = (
                "insufficient_history"
                if self.fit_method == "output_error"
                else "implausible_parameters"
            )
            self._keep_previous(d)
            return d
        parameters, flags, windows, readings, bridged = primary
        d.fit_method = self.fit_method
        alternative = (
            self._accepted_parameters(transition_fit)
            if self.fit_method == "output_error" and has_holdout
            else None
        )
        if alternative is not None:
            # Closed-loop zones (TRVs) can suit either estimator. Both are
            # cheap, so keep the one with the better held-out multi-step skill.
            # Choosing between two candidates biases the holdout only slightly.
            primary_score = self._skill_score(
                self._horizon_metrics(good[train_end:], parameters)
            )
            alternative_score = self._skill_score(
                self._horizon_metrics(good[train_end:], alternative)
            )
            if (
                alternative_score is not None
                and primary_score is not None
                and alternative_score < primary_score
            ):
                parameters, flags = alternative, ()
                d.fit_method = "transition"
        d.window_count = windows
        d.window_reading_count = readings
        d.bridged_hours = bridged
        d.identifiability_flags = list(flags)
        tau, gain, offset = parameters
        d.time_constant_hours = tau
        d.effective_gain_c_per_kw = gain
        d.equilibrium_offset_c = offset
        scored = slice(train_end, None) if has_holdout else slice(0, train_end)
        predicted = self._step(
            starts[scored],
            outdoors[scored],
            power_kw[scored],
            hours[scored],
            parameters,
        )
        errors = predicted - ends[scored]
        if not np.all(np.isfinite(errors)):
            d.unreliable_reason = "model_unstable"
            self._keep_previous(d)
            return d
        d.residual_p90_c = float(
            np.quantile(np.abs(errors), self.config.interval_coverage)
        )
        if has_holdout:
            d.holdout_count = holdout_count
            d.holdout_start_at = _utc(good[train_end].start_at)
            d.holdout_mae_c = float(np.mean(np.abs(errors)))
            d.holdout_bias_c = float(np.mean(errors))
            d.persistence_mae_c = float(
                np.mean(np.abs(starts[train_end:] - ends[train_end:]))
            )
            d.holdout_horizon_metrics = self._horizon_metrics(
                good[train_end:], parameters
            )
        evaluated = [
            d.holdout_horizon_metrics[key]
            for key in (f"{hour}h" for hour in self.config.skill_horizons_hours)
            if key in d.holdout_horizon_metrics
            and int(d.holdout_horizon_metrics[key]["count"])
            >= self.config.min_horizon_origins
        ]
        ratios = [
            float(item["mae_c"]) / max(float(item["persistence_mae_c"]), 1e-6)
            for item in evaluated
        ]
        d.skill_horizons_passed = sum(
            ratio <= 1 - self.config.min_persistence_improvement_ratio
            for ratio in ratios
        )
        biased = any(
            abs(float(item["bias_c"])) > self.config.max_holdout_bias_c
            for item in evaluated
        )
        if (
            gate_reason is None
            and not flags
            and not biased
            and d.skill_horizons_passed >= self.config.min_skill_horizons
        ):
            d.quality = "accepted"
            d.reliable_on_holdout = True
            d.unreliable_reason = None
        elif biased or (ratios and float(np.mean(ratios)) > 1.0):
            # Worse than holding the current reading: publishing it would be
            # worse than the naive fallback, so it is not a usable fit.
            d.unreliable_reason = gate_reason or "validation_failed"
            self._keep_previous(d)
            return d
        else:
            d.quality = "low_identifiability"
            d.unreliable_reason = gate_reason or (
                "low_identifiability" if flags else "validation_failed"
            )
        self._parameters = parameters
        d.fitted = True
        self.diagnostics = d
        return d

    def _skill_score(self, metrics: dict[str, dict[str, float | int]]) -> float | None:
        """Mean MAE relative to persistence over sufficiently sampled horizons."""
        ratios = [
            float(item["mae_c"]) / max(float(item["persistence_mae_c"]), 1e-6)
            for key, item in metrics.items()
            if int(key.removesuffix("h")) in self.config.skill_horizons_hours
            and int(item["count"]) >= self.config.min_horizon_origins
        ]
        return float(np.mean(ratios)) if ratios else None

    # The offline transition-fit diagnostics override this to compare the
    # previous one-step identification under the same acceptance pipeline.
    fit_method: Literal["output_error", "transition"] = "output_error"

    def _primary_fit(
        self, rows: list[ThermalTrainingTransition]
    ) -> tuple[tuple[float, float, float], tuple[str, ...], int, int, float] | None:
        """Return parameters, flags and window statistics for the primary fit."""
        if self.fit_method == "transition":
            parameters = self._fit(
                np.array([item.indoor_start_c for item in rows]),
                np.array([item.indoor_end_c for item in rows]),
                np.array([item.outdoor_mean_c for item in rows]),
                np.array([item.hvac_electric_mean_w / 1000 for item in rows]),
                np.array([item.elapsed_hours for item in rows]),
            )
            return None if parameters is None else (parameters, (), 0, 0, 0.0)
        windows = build_windows(
            self._chains(rows),
            min_hours=self.config.min_window_hours,
            max_hours=self.config.max_window_hours,
            stride_hours=self.config.window_stride_hours,
            max_bridge_fraction=self.config.max_bridge_fraction,
        )
        fit = fit_output_error(
            windows,
            min_tau=self.config.min_time_constant_hours,
            max_tau=self.config.max_time_constant_hours,
            grid_points=self.config.fit_grid_points,
            tau_prior_median=self.config.tau_prior_median_hours,
            tau_prior_log_sigma=self.config.tau_prior_log_sigma,
            gain_bounds=(
                self.config.min_effective_gain_c_per_kw,
                self.config.max_effective_gain_c_per_kw,
            ),
            offset_bound=self.config.max_equilibrium_offset_c,
            min_profile_relative_range=self.config.min_profile_relative_range,
        )
        if fit is None:
            return None
        return (
            fit.parameters,
            fit.flags,
            fit.window_count,
            fit.reading_count,
            windows.bridged_hours,
        )

    def _chains(
        self, rows: list[ThermalTrainingTransition]
    ) -> list[list[ChainSegment]]:
        """Link rows into contiguous simulation chains, bridging short gaps."""
        if not rows:
            return []
        epoch = _utc(rows[0].start_at)
        return build_chains(
            np.array(
                [(_utc(item.start_at) - epoch).total_seconds() / 3600 for item in rows]
            ),
            np.array(
                [(_utc(item.end_at) - epoch).total_seconds() / 3600 for item in rows]
            ),
            np.array([item.indoor_start_c for item in rows]),
            np.array([item.indoor_end_c for item in rows]),
            np.array([item.outdoor_mean_c for item in rows]),
            np.array([item.hvac_electric_mean_w / 1000 for item in rows]),
            max_bridge_hours=self.config.max_bridge_minutes / 60,
        )

    def evaluate_horizons(
        self,
        transitions: list[ThermalTrainingTransition],
        *,
        equilibrium_shift_c: float = 0.0,
    ) -> dict[str, dict[str, float | int]]:
        """Replay the active fit over weather-enriched real transitions."""
        if self._parameters is None:
            raise ValueError("thermal model is not fitted")
        tau, gain, offset = self._parameters
        ordered = sorted(transitions, key=lambda item: _utc(item.start_at))
        good: list[ThermalTrainingTransition] = []
        for item in ordered:
            if not good or _utc(item.start_at) >= _utc(good[-1].end_at):
                good.append(item)
        return self._horizon_metrics(good, (tau, gain, offset + equilibrium_shift_c))

    def _horizon_metrics(
        self,
        transitions: list[ThermalTrainingTransition],
        parameters: tuple[float, float, float],
    ) -> dict[str, dict[str, float | int]]:
        """Simulate from every real reading along bridged chains.

        These are response-model metrics (layer A), not operational forecast
        scores. Inputs across a bridged gap are filled; targets are only real
        readings, and a long gap terminates an origin.
        """
        targets = tuple(sorted({1, *self.config.skill_horizons_hours}))
        tolerance_hours = 0.25
        errors: dict[int, list[float]] = {hour: [] for hour in targets}
        baselines: dict[int, list[float]] = {hour: [] for hour in targets}
        tau, gain, offset = parameters
        horizon_limit = max(targets) + tolerance_hours
        for chain in self._chains(transitions):
            # Scalar recursion: origins times segments can reach 10^6 on a
            # full lookback of five-minute transitions.
            retain = [math.exp(-item.hours / tau) for item in chain]
            drive = [item.outdoor_c + gain * item.power_kw + offset for item in chain]
            for origin_index, first in enumerate(chain):
                initial = first.start_c
                state = initial
                elapsed = 0.0
                matched: set[int] = set()
                for index in range(origin_index, len(chain)):
                    item = chain[index]
                    decay = retain[index]
                    state = decay * state + (1 - decay) * drive[index]
                    elapsed += item.hours
                    for hour in targets:
                        if (
                            hour not in matched
                            and abs(elapsed - hour) <= tolerance_hours
                        ):
                            errors[hour].append(state - item.end_c)
                            baselines[hour].append(initial - item.end_c)
                            matched.add(hour)
                    if elapsed > horizon_limit:
                        break
        metrics: dict[str, dict[str, float | int]] = {}
        for hour in targets:
            if not errors[hour]:
                continue
            residuals = np.array(errors[hour])
            persistence = np.array(baselines[hour])
            metrics[f"{hour}h"] = {
                "count": len(residuals),
                "mae_c": float(np.mean(np.abs(residuals))),
                "bias_c": float(np.mean(residuals)),
                "rmse_c": float(np.sqrt(np.mean(residuals**2))),
                "persistence_mae_c": float(np.mean(np.abs(persistence))),
                "residual_p90_c": float(
                    np.quantile(np.abs(residuals), self.config.interval_coverage)
                ),
            }
        return metrics

    def _fit(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        outdoors: np.ndarray,
        power_kw: np.ndarray,
        hours: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Search bounded time constants; solve remaining coefficients by LS."""
        candidate = fit_candidate(
            starts,
            ends,
            outdoors,
            power_kw,
            hours,
            min_tau=self.config.min_time_constant_hours,
            max_tau=self.config.max_time_constant_hours,
            grid_points=self.config.fit_grid_points,
        )
        return self._accepted_parameters(candidate)

    def _accepted_parameters(
        self, candidate: FitCandidate | None
    ) -> tuple[float, float, float] | None:
        """Apply the unchanged physical gates to runtime or diagnostic fits."""
        best = candidate.parameters if candidate is not None else None
        # A boundary optimum is weakly identified and is not made reliable by
        # silently clipping the fitted time constant.
        if best is None or best[0] in (
            self.config.min_time_constant_hours,
            self.config.max_time_constant_hours,
        ):
            return None
        if not (
            math.isfinite(best[1])
            and math.isfinite(best[2])
            and 0 < best[1] <= self.config.max_effective_gain_c_per_kw
            and abs(best[2]) <= self.config.max_equilibrium_offset_c
        ):
            return None
        return best

    @staticmethod
    def _step(
        indoor: np.ndarray,
        outdoor: np.ndarray,
        power_kw: np.ndarray,
        hours: np.ndarray,
        parameters: tuple[float, float, float],
    ) -> np.ndarray:
        """Evaluate the exact constant-input first-order discrete solution."""
        tau, gain, offset = parameters
        decay = np.exp(-hours / tau)
        return cast(
            np.ndarray,
            decay * indoor + (1 - decay) * (outdoor + gain * power_kw + offset),
        )

    def predict(
        self,
        *,
        origin: datetime,
        initial_temperature_c: float,
        outdoor_temperature_c: list[float],
        hvac_electric_power_w: list[float],
        outdoor_source: str,
        hvac_drive_source: str,
        equilibrium_shift_c: float = 0.0,
    ) -> ThermalPrediction:
        """Simulate an exact UTC future grid from explicit outdoor and HVAC drive.

        Args:
            origin: Aligned start of the first future drive interval.
            initial_temperature_c: Real latest indoor observation, freshness
                checked by the caller.
            outdoor_temperature_c: Outdoor temperature for every interval.
            hvac_electric_power_w: Forecast electrical mean W per interval.
            outdoor_source: Provenance label for weather forecast.
            hvac_drive_source: Provenance label for electrical forecast.
            equilibrium_shift_c: Constant offset added to the fitted
                equilibrium, e.g. a zone's mean offset from the aggregate.

        Returns:
            Temperature trajectory with horizon-widening empirical bounds.

        Raises:
            ValueError: When model, grid, or any required input is invalid.
        """
        if self._parameters is None:
            raise ValueError("thermal model is not fitted")
        origin_utc = _utc(origin)
        step_seconds = self.config.interval_minutes * 60
        if origin_utc.timestamp() % step_seconds != 0:
            raise ValueError("thermal prediction origin is off the UTC grid")
        periods = len(hvac_electric_power_w)
        if not 0 < periods <= self.config.max_prediction_periods:
            raise ValueError("thermal prediction period count is invalid")
        if len(outdoor_temperature_c) != periods:
            raise ValueError("outdoor and HVAC drive grids differ")
        if not outdoor_source or not hvac_drive_source:
            raise ValueError("outdoor and HVAC provenance are required")
        state = float(initial_temperature_c)
        if not math.isfinite(state) or not math.isfinite(equilibrium_shift_c):
            raise ValueError("initial temperature and shift must be finite")
        scale = self.diagnostics.residual_p90_c
        if scale is None or not math.isfinite(scale):
            raise ValueError("empirical uncertainty is unavailable")
        widen = (
            1.0
            if self.diagnostics.quality == "accepted"
            else self.config.low_identifiability_interval_scale
        )
        tau, gain, offset = self._parameters
        parameters = (tau, gain, offset + equilibrium_shift_c)
        intervals: list[ThermalPredictionInterval] = []
        for index, (outdoor, power_w) in enumerate(
            zip(outdoor_temperature_c, hvac_electric_power_w, strict=True), start=1
        ):
            if not math.isfinite(outdoor) or not math.isfinite(power_w) or power_w < 0:
                raise ValueError("future weather and HVAC drive must be finite")
            state = float(
                self._step(
                    np.array([state]),
                    np.array([outdoor]),
                    np.array([power_w / 1000]),
                    np.array([self.config.interval_minutes / 60]),
                    parameters,
                )[0]
            )
            if not math.isfinite(state):
                raise ValueError("thermal prediction became non-finite")
            # Random-walk residual accumulation is conservative for an initial
            # empirical interval, pending real-origin horizon calibration.
            horizon_hours = index * self.config.interval_minutes / 60
            empirical = max(
                (
                    float(item["residual_p90_c"])
                    for key, item in self.diagnostics.holdout_horizon_metrics.items()
                    if int(key.removesuffix("h")) <= horizon_hours
                ),
                default=0.0,
            )
            width = widen * max(scale * math.sqrt(index), empirical)
            intervals.append(
                ThermalPredictionInterval(
                    timestamp=origin_utc + timedelta(seconds=index * step_seconds),
                    indoor_temperature_c=state,
                    lower_temperature_c=state - width,
                    upper_temperature_c=state + width,
                )
            )
        return ThermalPrediction(
            origin=origin_utc,
            outdoor_source=outdoor_source,
            hvac_drive_source=hvac_drive_source,
            intervals=intervals,
        )

    def save(self) -> None:
        """Atomically publish a complete compatible model manifest.

        Raises:
            ValueError: When no valid fit is available.
        """
        if self._parameters is None:
            raise ValueError("cannot save an unfitted thermal model")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest_path()
        temporary = manifest.with_suffix(".json.tmp")
        payload = {
            "contract_version": THERMAL_CONTRACT_VERSION,
            "package_version": __version__,
            "model_id": self.model_id,
            "authority_fingerprint": self.authority_fingerprint,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "weather_config": (
                self.weather_client.config.model_dump(mode="json")
                if self.weather_client is not None
                else None
            ),
            "config": self.config.model_dump(mode="json"),
            "parameters": self._parameters,
            "diagnostics": self.diagnostics.model_dump(mode="json"),
        }
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest)

    def load(self) -> bool:
        """Restore a complete artifact of the same lineage and contract.

        Corrupt or incompatible data are rejected without becoming active. An
        artifact written by another package version or weather configuration
        is restored with ``needs_retrain`` set: those are compatibility inputs,
        so the caller retrains on the same history instead of discarding the
        published fit.
        """
        self._parameters = None
        self.needs_retrain = False
        self.diagnostics = ThermalModelDiagnostics()
        manifest = self._manifest_path()
        if not manifest.exists():
            return False
        try:
            with manifest.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if (
                payload["contract_version"] != THERMAL_CONTRACT_VERSION
                or payload["model_id"] != self.model_id
                or payload["authority_fingerprint"] != self.authority_fingerprint
                or payload["latitude"] != self.latitude
                or payload["longitude"] != self.longitude
                or payload["config"] != self.config.model_dump(mode="json")
            ):
                return False
            needs_retrain = payload["package_version"] != __version__ or payload[
                "weather_config"
            ] != (
                self.weather_client.config.model_dump(mode="json")
                if self.weather_client is not None
                else None
            )
            parameters = tuple(float(value) for value in payload["parameters"])
            if len(parameters) != 3:
                return False
            tau, gain, offset = parameters
            if not (
                math.isfinite(tau)
                and math.isfinite(gain)
                and math.isfinite(offset)
                and self.config.min_time_constant_hours
                <= tau
                <= self.config.max_time_constant_hours
                and 0 < gain <= self.config.max_effective_gain_c_per_kw
                and abs(offset) <= self.config.max_equilibrium_offset_c
            ):
                return False
            diagnostics = ThermalModelDiagnostics.model_validate(payload["diagnostics"])
            if not diagnostics.fitted or diagnostics.quality is None:
                return False
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        self._parameters = (tau, gain, offset)
        self.diagnostics = diagnostics
        self.needs_retrain = needs_retrain
        return True

    def _manifest_path(self) -> Path:
        """Return the model-specific completion manifest path."""
        # Avoid path traversal while keeping the stable public model identifier.
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.model_id)
        suffix = sha256(self.model_id.encode()).hexdigest()[:12]
        return self.storage_path / f"{safe_id}_{suffix}_thermal_manifest.json"
