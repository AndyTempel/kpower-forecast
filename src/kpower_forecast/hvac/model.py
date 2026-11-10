"""Candidate fitting, explicit installation, causal prediction and atomic artifacts.

The model owns no background jobs, network clients or active control state. A
caller can discard a late FitResult without altering the installed forecast.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, RLock
from time import monotonic
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue

from kpower_forecast import __version__

from ._stats import censored_lognormal_aft, check_budget
from .aggregate import _integral, interval_stats, representatives
from .cycle_model import CycleFit, fit_cycles
from .cycles import clean_running, extract_cycles, modes_at
from .energy import EnergyFit, energy_paths, fit_energy
from .frames import (
    history_verdict,
    outdoor_frame,
    prepare_minutes,
    running_states,
    weather_minutes,
)
from .outdoor import OutdoorFit, fit_outdoor, outdoor_paths
from .power import PowerFit, fit_power
from .simulate import initial_state, schedule_modes, simulate
from .types import (
    HVAC_CONTRACT_VERSION,
    HVAC_ESTIMATOR_VERSION,
    FitResult,
    HistoryPolicy,
    HvacCapabilities,
    HvacError,
    HvacFamily,
    HvacForecast,
    HvacForecastRequest,
    HvacModelConfig,
    HvacTrainingData,
    InputTier,
    utc,
)


class _State(BaseModel):
    """Validated coefficients/calibration only; no minute paths or training frames."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )
    config: HvacModelConfig
    capabilities: HvacCapabilities
    policy: HistoryPolicy
    lineage: dict[str, str]
    outdoor: OutdoorFit
    cycle: CycleFit | None = None
    power: PowerFit | None = None
    energy: EnergyFit | None = None
    threshold_w: float | None = None
    comfort_off: dict[str, tuple[float, float]]
    outdoor_low_c: float
    outdoor_high_c: float


class _Artifact(BaseModel):
    """Versioned complete artifact, published by atomic replacement."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    contract: int
    estimator: str
    package_version: str
    fitted_at: AwareDatetime
    fingerprint: str
    tier: InputTier
    family: HvacFamily
    fallback_reason: str | None
    parameters: _State
    diagnostics: dict[str, JsonValue]


def _fingerprint(
    cfg: HvacModelConfig,
    capabilities: HvacCapabilities,
    policy: HistoryPolicy,
    lineage: Mapping[str, str],
) -> str:
    config_identity = cfg.model_dump(mode="json")
    policy_identity = policy.model_dump(mode="json")
    config_identity["off_modes"] = sorted(cfg.off_modes)
    policy_identity["measured_valid_qualities"] = sorted(
        policy.measured_valid_qualities
    )
    identity = {
        "contract": HVAC_CONTRACT_VERSION,
        "estimator": HVAC_ESTIMATOR_VERSION,
        "config": config_identity,
        "capabilities": capabilities.model_dump(mode="json"),
        "policy": policy_identity,
        "lineage": dict(lineage),
        "outdoor_precedence": ["unit", "environment", "weather"],
        "tier_set": ["energy", "power"],
    }
    # These inputs are sets, not ordered statistical quantities. Their JSON
    # iteration order can change on a validation round-trip or another process.
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def fingerprint(data: HvacTrainingData, cfg: HvacModelConfig) -> str:
    """Compute a compatibility identity without fitting or touching the active model.

    Args:
        data: Capability, policy and source lineage snapshot.
        cfg: Public tunables and timezone.

    Returns:
        SHA-256 identity including contract, tier set and outdoor precedence.
    """
    return _fingerprint(cfg, data.capabilities, data.history_policy, data.lineage)


def _events_at(events: pd.DataFrame, origin: datetime) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if "time" not in events:
        raise HvacError("event history requires time")
    time = pd.DatetimeIndex(events.time)
    if str(time.tz) != "UTC":
        raise HvacError("event history requires aware UTC")
    mask = time <= pd.Timestamp(origin)
    if "available_at" in events:
        availability = pd.DatetimeIndex(events.available_at)
        if str(availability.tz) != "UTC":
            raise HvacError("event availability requires aware UTC")
        mask &= availability <= pd.Timestamp(origin)
    return cast(
        pd.DataFrame, events.loc[mask].copy().sort_values("time", kind="stable")
    )


def _weather_at(
    weather: pd.DataFrame, origin: datetime, *, history: bool
) -> pd.DataFrame:
    result = weather.copy()
    if "available_at" in result:
        available = pd.DatetimeIndex(result.available_at)
        if str(available.tz) != "UTC":
            raise HvacError("weather availability requires aware UTC")
        result = cast(pd.DataFrame, result.loc[available <= pd.Timestamp(origin)])
    if history:
        stamp = pd.DatetimeIndex(result.ds if "ds" in result else result.index)
        if str(stamp.tz) != "UTC":
            raise HvacError("weather timestamps require aware UTC")
        result = cast(pd.DataFrame, result.loc[stamp <= pd.Timestamp(origin)])
    return result


def _comfort_lengths(
    events: pd.DataFrame,
    cfg: HvacModelConfig,
    as_of: datetime,
    cancel: Event | None,
    deadline: float,
) -> dict[str, tuple[float, float]]:
    durations: dict[str, list[tuple[float, bool]]] = {}
    active: tuple[datetime, str] | None = None
    for row in events.itertuples():
        when = cast(pd.Timestamp, row.time).to_pydatetime()
        mode = str(row.mode)
        if active is not None and mode not in cfg.off_modes:
            length = (when - active[0]).total_seconds() / 60
            if length > 0:
                durations.setdefault(active[1], []).append((length, False))
            active = None
        if active is None and mode in cfg.off_modes:
            active = when, f"{mode}|{row.source}"
    if active is not None:
        length = (as_of - active[0]).total_seconds() / 60
        if length > 0:
            durations.setdefault(active[1], []).append((length, True))
    fitted = {}
    for context, spells in durations.items():
        if sum(not censored for _, censored in spells) < 3:
            continue
        fit = censored_lognormal_aft(
            np.ones((len(spells), 1)),
            np.array([v[0] for v in spells]),
            np.array([v[1] for v in spells]),
            cancel=cancel,
            deadline=deadline,
        )
        if not fit.converged:
            raise HvacError("comfort_off_aft_nonconvergence")
        fitted[context] = float(fit.coefficients[0]), fit.sigma
    return fitted


def _fit_candidate(
    data: HvacTrainingData,
    cfg: HvacModelConfig,
    *,
    as_of: datetime,
    cancel: Event | None = None,
    force_energy: bool = False,
) -> FitResult:
    as_of = utc(as_of)
    began = monotonic()
    deadline = began + 120.0
    check_budget(cancel, deadline)
    frame = prepare_minutes(
        data.minutes,
        cfg,
        as_of=as_of,
        policy=data.history_policy,
        epoch_start=data.epoch_start,
    )
    if frame.empty:
        raise HvacError("no_eligible_history")
    events = _events_at(data.mode_events, as_of)
    control = _events_at(data.control_events, as_of)
    weather = _weather_at(data.weather, as_of, history=True)
    frame = outdoor_frame(frame, weather, cfg)
    frame["mode"] = modes_at(pd.DatetimeIndex(frame.index), events, cfg.reference_mode)
    quality = history_verdict(frame, data.history_policy, as_of=as_of)
    diagnostic: dict[str, JsonValue] = {
        **asdict(quality),
        "history_policy_version": data.history_policy.version,
        "history_policy_settings": data.history_policy.settings,
        "outdoor_source_counts": {
            str(k): int(v) for k, v in frame.outdoor_source.value_counts().items()
        },
    }
    if "regime" in frame:
        excluded = ~frame.regime.isin(
            ["space_heating", "standby", "defrost", "unknown"]
        )
        diagnostic["non_heating_observed_kwh"] = float(
            frame.loc[excluded, "power_w"].sum() / 60000
        )
        frame.loc[excluded, "power_w"] = np.nan
    classified = running_states(frame, data.capabilities.run_threshold_w)
    frame["running"] = clean_running(
        classified.running,
        max_dip_min=cfg.dip_merge_min,
        min_run_min=cfg.min_run_blip_min,
        max_gap_min=cfg.minute_gap_tolerance_min,
    )
    family = data.capabilities.family
    fallback: str | None = "energy_only_requested" if force_energy else None
    if family in (HvacFamily.TWO_STAGE, HvacFamily.MULTI_STAGE, HvacFamily.INVERTER):
        fallback = f"family_not_implemented:{family.value}"
    elif family is HvacFamily.UNKNOWN:
        if classified.reason is None:
            family = HvacFamily.FIXED_SPEED
            diagnostic["family_inferred_fixed_speed"] = True
        else:
            fallback = classified.reason
    elif classified.reason:
        fallback = classified.reason
    cycle = None
    power = None
    energy = None
    tier = InputTier.ENERGY
    outdoor = fit_outdoor(frame, weather, cfg.timezone)
    if fallback is None:
        extraction = extract_cycles(frame, events, cfg, control_events=control)
        sojourns = extraction.sojourns
        cycle, fallback, flags = fit_cycles(
            sojourns, cfg, as_of=as_of, cancel=cancel, deadline=deadline
        )
        diagnostic.update(flags)
        diagnostic["complete_runs"] = int(
            (sojourns.kind.eq("run") & ~sojourns.censored).sum()
        )
        diagnostic["complete_idles"] = int(
            (sojourns.kind.eq("idle") & ~sojourns.censored).sum()
        )
        diagnostic["censored_share"] = (
            float(sojourns.censored.mean()) if len(sojourns) else 0.0
        )
        diagnostic["dropped_left_edges"] = extraction.dropped_left_edges
        if cycle is not None:
            power = fit_power(frame, events, cfg, data.capabilities.metered_scope)
            tier = InputTier.POWER
            diagnostic.update(
                {
                    "idle_sigma": cycle.idle_sigma,
                    "run_sigma": cycle.run_sigma,
                    "run_power_w_at_7c_minute_10": float(power.run_power(10, 7.0)),
                    "run_power_w_at_7c_minute_20": float(power.run_power(20, 7.0)),
                    "idle_power_w": power.idle_w,
                }
            )
            for temperature in (-15, -10, 0, 7, 15):
                duty, idle_min, run_min = cycle.stationary(
                    float(temperature), cfg.reference_mode
                )
                diagnostic[f"duty_at_{temperature}c"] = duty
                diagnostic[f"idle_median_min_at_{temperature}c"] = idle_min
                diagnostic[f"run_median_min_at_{temperature}c"] = run_min
    if cycle is None:
        energy = fit_energy(frame, events, cfg, cancel=cancel, deadline=deadline)
    off = _comfort_lengths(events, cfg, as_of, cancel, deadline)
    known = frame.outdoor_c.dropna()
    if known.empty:
        raise HvacError("no_outdoor_history")
    state = _State(
        config=cfg,
        capabilities=data.capabilities,
        policy=data.history_policy,
        lineage=dict(data.lineage),
        outdoor=outdoor,
        cycle=cycle,
        power=power,
        energy=energy,
        threshold_w=classified.threshold_w,
        comfort_off=off,
        outdoor_low_c=float(known.quantile(0.02)),
        outdoor_high_c=float(known.quantile(0.98)),
    )
    diagnostic.update(
        {
            "tier": tier.value,
            "family": family.value,
            "fallback_reason": fallback,
            "mos_on_reanalysis": outdoor.on_reanalysis,
            "mos_lead_unaware": outdoor.lead_unaware,
            "mos_weather_reference": outdoor.weather_reference,
            "fit_seconds": monotonic() - began,
            "comfort_loop_not_simulated": True,
        }
    )
    check_budget(cancel, deadline)
    parameters = cast(dict[str, JsonValue], state.model_dump(mode="json"))
    json.dumps(parameters, allow_nan=False)
    return FitResult(
        as_of,
        fingerprint(data, cfg),
        tier,
        family,
        MappingProxyType(parameters),
        MappingProxyType(diagnostic),
        fallback,
    )


def _validated_state(candidate: FitResult, cfg: HvacModelConfig) -> _State:
    if candidate.estimator_version != HVAC_ESTIMATOR_VERSION:
        raise HvacError("incompatible HVAC estimator")
    state = _State.model_validate(dict(candidate.parameters))
    if state.config != cfg or candidate.fingerprint != _fingerprint(
        cfg, state.capabilities, state.policy, state.lineage
    ):
        raise HvacError("incompatible HVAC configuration or fingerprint")
    if (state.cycle is None) != (candidate.tier is InputTier.ENERGY):
        raise HvacError("candidate tier disagrees with coefficients")
    # CycleFit and PowerFit validate their own coefficients on construction.
    if state.cycle is not None and state.power is None:
        raise HvacError("incomplete fixed-speed coefficients")
    if state.cycle is not None and candidate.family is not HvacFamily.FIXED_SPEED:
        raise HvacError("cycle coefficients require fixed-speed family")
    if state.cycle is None and state.energy is None:
        raise HvacError("missing energy coefficients")
    if state.energy is not None:
        energy = state.energy
        if (
            len(energy.coefficients) != 3 + len(energy.modes)
            or not energy.residual_profiles_w
            or any(len(row) != 60 for row in energy.residual_profiles_w)
        ):
            raise HvacError("invalid energy calibration dimensions")
    json.dumps(dict(candidate.parameters), allow_nan=False)
    utc(candidate.fitted_at)
    return state


def _predict_candidate(
    request: HvacForecastRequest,
    candidate: FitResult,
    cfg: HvacModelConfig,
    *,
    cancel: Event | None = None,
    force_phase_unknown: bool = False,
) -> HvacForecast:
    began = monotonic()
    deadline = began + request.deadline_s
    check_budget(cancel, deadline)
    state = _validated_state(candidate, cfg)
    if request.origin < candidate.fitted_at:
        raise HvacError("forecast origin precedes the fit availability")
    if request.grid[0].start < request.origin:
        raise HvacError("forecast grid cannot precede origin")
    start = pd.Timestamp(request.origin).floor("min").to_pydatetime()
    horizon = int(np.ceil((request.grid[-1].end - start).total_seconds() / 60))
    index = pd.date_range(start, periods=horizon, freq="min")
    rng = np.random.default_rng(cfg.seed ^ (int(request.origin.timestamp()) // 60))
    weather = _weather_at(request.weather_forecast, request.origin, history=False)
    if "ds" in weather:
        weather = weather.set_index("ds")
    recent = prepare_minutes(
        request.recent,
        cfg,
        as_of=request.origin,
        policy=state.policy,
        training=False,
    )
    if recent.empty:
        recent = pd.DataFrame({"power_w": [np.nan]}, index=pd.DatetimeIndex([start]))
    recent = outdoor_frame(recent, weather, cfg)
    classified = running_states(recent, state.threshold_w)
    recent["running"] = clean_running(
        classified.running,
        max_dip_min=cfg.dip_merge_min,
        min_run_min=cfg.min_run_blip_min,
        max_gap_min=cfg.minute_gap_tolerance_min,
    )
    # Anchor on a sensor only when it is the current source and its original
    # reading (not the held copy) is within the hold window of the origin.
    observed = None
    source = str(recent.outdoor_source.iloc[-1])
    raw = {"unit": "outdoor_unit_c", "environment": "outdoor_env_c"}.get(source)
    if raw is not None:
        readings = recent[raw].dropna()
        if len(readings) and request.origin - readings.index[
            -1
        ].to_pydatetime() <= timedelta(minutes=cfg.outdoor_hold_min):
            observed = float(recent.outdoor_c.iloc[-1])
    temperatures = outdoor_paths(
        state.outdoor,
        weather,
        index,
        origin=request.origin,
        timezone=cfg.timezone,
        n_paths=cfg.n_paths,
        rng=rng,
        observed_c=observed,
    )
    covariates = weather_minutes(weather, index)
    sun = covariates.shortwave_radiation.to_numpy(dtype=float) / 1000
    if not np.isfinite(sun).all():
        raise HvacError("weather does not cover radiation horizon")
    comfort = None
    for span in request.schedule:
        if span.end_uncertain:
            found = state.comfort_off.get(f"{span.mode}|{span.source}")
            if comfort is not None and found != comfort:
                raise HvacError("multiple uncertain comfort contexts")
            comfort = found
    modes, assumptions = schedule_modes(request, cfg, index, rng, comfort_off=comfort)
    diagnostic: dict[str, JsonValue] = dict(candidate.diagnostics)
    diagnostic["n_paths"] = cfg.n_paths
    diagnostic["run_threshold_w"] = state.threshold_w
    diagnostic.update({key: True for key in assumptions})
    diagnostic["outdoor_extrapolated"] = bool(
        np.min(temperatures) < state.outdoor_low_c - 2
    )
    if state.cycle is None:
        assert state.energy is not None
        mean = state.energy.mean_power(
            temperatures.astype(float),
            np.broadcast_to(sun, temperatures.shape),
            modes.names(),
        )
        paths = energy_paths(state.energy, mean, rng, start)
        shown = np.empty((0, len(request.grid)), np.float32)
        shown.setflags(write=False)
        display = None
        tier = InputTier.ENERGY
    else:
        assert state.power is not None
        phase = (
            initial_state(recent.iloc[:0], request.origin, cfg)
            if force_phase_unknown
            else initial_state(recent, request.origin, cfg)
        )
        if force_phase_unknown:
            # Phase ablation hides compressor-state evidence; covariates stay identical.
            diagnostic["phase_forced_unknown"] = True
        simulated = simulate(
            request,
            state.cycle,
            state.power,
            phase,
            cfg,
            state.capabilities,
            temperatures,
            sun,
            modes,
            rng,
            cancel=cancel,
            deadline=deadline,
        )
        paths = simulated.paths
        shown, display = representatives(paths, request.grid)
        tier = InputTier.POWER
        diagnostic.update({key: True for key in simulated.diagnostics})
        diagnostic["stationary_initial_state"] = phase.running is None
    intervals = interval_stats(paths, request.grid)
    offset = (request.origin - paths.start).total_seconds() / 60
    horizons = [
        hours for hours in (1, 6, 24) if offset + hours * 60 <= paths.power_w.shape[1]
    ]
    if horizons:
        # Means are additive. Retain original actual-origin totals without
        # serializing paths or relying on an upcoming-quarter display grid.
        mean_power = np.mean(paths.power_w, axis=0, dtype=float)
        energy = (
            _integral(
                mean_power[None, :],
                np.full(len(horizons), offset),
                offset + np.asarray(horizons) * 60,
            )[0]
            / 60000
        )
        diagnostic["horizon_energy_kwh"] = {
            str(hours): float(value)
            for hours, value in zip(horizons, energy, strict=True)
        }
    diagnostic.update({"tier": tier.value, "predict_seconds": monotonic() - began})
    check_budget(cancel, deadline)
    return HvacForecast(
        request.origin,
        request.grid,
        intervals,
        paths,
        shown,
        display,
        tier,
        candidate.family,
        candidate.fallback_reason,
        MappingProxyType(diagnostic),
        candidate.fingerprint,
    )


class KPowerHvacForecast:
    """Thread-safe active candidate, with explicit fit/install/save boundaries."""

    def __init__(
        self, config: HvacModelConfig, *, storage_dir: Path | None = None
    ) -> None:
        """Create an empty producer without fitting, loading or creating files.

        Args:
            config: HVAC tunables and local timezone.
            storage_dir: Optional artifact directory, created only on save.
        """
        self.config = config
        self.storage_dir = storage_dir
        self._active: FitResult | None = None
        self._lock = RLock()

    @property
    def active_fit(self) -> FitResult | None:
        """Return the installed candidate snapshot, or None."""
        with self._lock:
            if self._active is None:
                return None
            candidate = self._active
            return FitResult(
                candidate.fitted_at,
                candidate.fingerprint,
                candidate.tier,
                candidate.family,
                MappingProxyType(json.loads(json.dumps(dict(candidate.parameters)))),
                MappingProxyType(json.loads(json.dumps(dict(candidate.diagnostics)))),
                candidate.fallback_reason,
                candidate.estimator_version,
            )

    def fit(
        self, data: HvacTrainingData, *, as_of: datetime, cancel: Event | None = None
    ) -> FitResult:
        """Return a candidate; never install it, create directories or save files.

        Args:
            data: Evidence and lineage snapshot.
            as_of: Causal fit origin.
            cancel: Cooperative cancellation.

        Returns:
            Validated candidate with an independent reliability verdict.
        """
        return _fit_candidate(data, self.config, as_of=as_of, cancel=cancel)

    def install(self, fit_result: FitResult) -> None:
        """Atomically install an isolated, compatible candidate without saving it.

        Args:
            fit_result: Candidate accepted by the caller's generation/deadline guard.

        Raises:
            ValueError: If its artifact/configuration is incompatible or incomplete.
        """
        state = _validated_state(fit_result, self.config)
        copied = FitResult(
            fit_result.fitted_at,
            fit_result.fingerprint,
            fit_result.tier,
            fit_result.family,
            MappingProxyType(cast(dict[str, JsonValue], state.model_dump(mode="json"))),
            MappingProxyType(
                json.loads(json.dumps(dict(fit_result.diagnostics), allow_nan=False))
            ),
            fit_result.fallback_reason,
        )
        with self._lock:
            self._active = copied

    def predict(
        self, request: HvacForecastRequest, *, cancel: Event | None = None
    ) -> HvacForecast:
        """Predict from one installed snapshot without changing it.

        Args:
            request: Causal recent evidence and weather/schedule available at origin.
            cancel: Cooperative cancellation.

        Returns:
            Forecast retaining every minute path in memory.

        Raises:
            ValueError: If no candidate is installed or inputs are incompatible.
        """
        candidate = self.active_fit
        if candidate is None:
            raise HvacError("no installed HVAC fit")
        return _predict_candidate(request, candidate, self.config, cancel=cancel)

    @property
    def artifact_path(self) -> Path:
        """Return a safe model-id artifact filename inside the configured directory."""
        if self.storage_dir is None:
            raise HvacError("artifact directory is not configured")
        identifier = self.config.model_id
        if (
            Path(identifier).name != identifier
            or identifier in {".", ".."}
            or "\\" in identifier
        ):
            raise HvacError("model_id must be a safe filename component")
        return self.storage_dir / f"hvac_{identifier}.json"

    def save(self) -> None:
        """Atomically persist the installed coefficients, never the path ensemble.

        Raises:
            ValueError: If no candidate/directory is configured.
            OSError: If writing or replacing the complete artifact fails.
        """
        candidate = self.active_fit
        if candidate is None:
            raise HvacError("no installed HVAC fit")
        state = _validated_state(candidate, self.config)
        artifact = _Artifact(
            contract=HVAC_CONTRACT_VERSION,
            estimator=HVAC_ESTIMATOR_VERSION,
            package_version=__version__,
            fitted_at=candidate.fitted_at,
            fingerprint=candidate.fingerprint,
            tier=candidate.tier,
            family=candidate.family,
            fallback_reason=candidate.fallback_reason,
            parameters=state,
            diagnostics=dict(candidate.diagnostics),
        )
        destination = self.artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(artifact.model_dump_json())
            temporary.flush()
            os.fsync(temporary.fileno())
            path = Path(temporary.name)
        os.replace(path, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def load(self, *, expected_fingerprint: str) -> bool:
        """Install a complete matching artifact, preserving active state on failure.

        Args:
            expected_fingerprint: Current capability, policy and lineage identity.

        Returns:
            False for missing/corrupt/incompatible files; True after installation.
        """
        try:
            artifact = _Artifact.model_validate_json(self.artifact_path.read_bytes())
            if (
                artifact.contract != HVAC_CONTRACT_VERSION
                or artifact.estimator != HVAC_ESTIMATOR_VERSION
                or artifact.fingerprint != expected_fingerprint
            ):
                return False
            candidate = FitResult(
                artifact.fitted_at,
                artifact.fingerprint,
                artifact.tier,
                artifact.family,
                cast(dict[str, JsonValue], artifact.parameters.model_dump(mode="json")),
                artifact.diagnostics,
                artifact.fallback_reason,
                artifact.estimator,
            )
            self.install(candidate)
        except (OSError, ValueError, TypeError, OverflowError):
            return False
        return True
