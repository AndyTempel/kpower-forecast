"""Event-batched fixed-speed simulation; no network or active-model installation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import monotonic

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._stats import check_budget, conditional_lognormal
from .cycle_model import OVERLOAD_MARGIN, CycleFit
from .initial_state import InitialState
from .power import PowerFit
from .types import (
    HvacCapabilities,
    HvacError,
    HvacForecastRequest,
    HvacModelConfig,
    MinutePaths,
)


@dataclass(frozen=True)
class SimulationResult:
    """Full retained ensemble and diagnostics, with no display-path reduction."""

    paths: MinutePaths
    diagnostics: tuple[str, ...]


def schedule_modes(
    request: HvacForecastRequest,
    cfg: HvacModelConfig,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    *,
    comfort_off: tuple[float, float] | None = None,
) -> tuple[NDArray[np.str_], tuple[str, ...]]:
    """Project known modes and one conditional release per active comfort latch.

    Args:
        request: Nonoverlapping effective spans, including planned release modes.
        cfg: Modes, path count and reference mode.
        index: Minute starts covering the future horizon.
        rng: Prediction-owned deterministic generator.
        comfort_off: Learned log-normal OFF duration, required for uncertain spans.

    Returns:
        Mode per path/minute and explicit assumption diagnostics. Split uncertain
        spans share one sampled release, preserving their planned release modes.

    Raises:
        ValueError: If uncertain release has no learned distribution or resume mode.
    """
    labels = {cfg.reference_mode, request.last_mode or cfg.reference_mode}
    for span in request.schedule:
        labels.add(span.mode)
        if span.release_mode:
            labels.add(span.release_mode)
    width = max(map(len, labels))
    modes = np.full(
        (cfg.n_paths, len(index)),
        request.last_mode or cfg.reference_mode,
        dtype=f"<U{width}",
    )
    diagnostics = [] if request.last_mode else ["mode_assumed_reference"]
    release: NDArray[np.float64] | None = None
    for span in request.schedule:
        start = max(
            int(np.ceil((span.start - index[0].to_pydatetime()).total_seconds() / 60)),
            0,
        )
        end = min(
            int(np.ceil((span.end - index[0].to_pydatetime()).total_seconds() / 60)),
            len(index),
        )
        if start >= len(index) or end <= 0:
            continue
        if not span.end_uncertain:
            modes[:, start:end] = span.mode
            continue
        if comfort_off is None or span.release_mode is None:
            raise HvacError("uncertain_comfort_release_unidentified")
        if span.mode not in cfg.off_modes:
            raise HvacError("uncertain span must be an off mode")
        if release is None:
            start_offset = (span.start - request.origin).total_seconds() / 60
            age_at_origin = max(span.elapsed_min - start_offset, 0.0)
            remaining = conditional_lognormal(
                comfort_off[0],
                comfort_off[1],
                np.full(cfg.n_paths, age_at_origin),
                rng,
            )
            release = (
                request.origin - index[0].to_pydatetime()
            ).total_seconds() / 60 + remaining
        at = np.arange(start, end)
        modes[:, start:end] = np.where(
            at[None, :] < release[:, None], span.mode, span.release_mode
        )
    return modes, tuple(diagnostics)


class _Rates:
    """Packed per-path parameters, avoiding a Python loop over paths or minutes."""

    def __init__(
        self, fit: CycleFit, cfg: HvacModelConfig, labels: NDArray[np.str_]
    ) -> None:
        selected = np.arange(cfg.n_paths) % len(fit.parameter_sets)
        parameters = [fit.parameter_sets[i] for i in selected]
        self.tbal = np.array([p.t_bal_c for p in parameters])
        self.rho = np.array([p.rho for p in parameters])
        self.slope = np.array([p.capacity_slope for p in parameters])
        self.si = (
            np.array([p.sigma_idle for p in parameters]) * cfg.dispersion_inflation
        )
        self.sr = np.array([p.sigma_run for p in parameters]) * cfg.dispersion_inflation
        self.recovery = np.array([p.recovery_multiplier for p in parameters])
        self.beta = np.array([p.idle_beta[0] for p in parameters])
        self.weather = np.array([p.idle_beta[-3:] for p in parameters])
        self.idle = np.array(
            [
                [
                    sum(
                        b
                        for m, b in zip(p.idle_modes, p.idle_beta[1:-3], strict=True)
                        if m == label
                    )
                    for label in labels
                ]
                for p in parameters
            ]
        )
        self.run = np.array(
            [
                [
                    sum(
                        b
                        for m, b in zip(p.run_modes, p.run_beta, strict=True)
                        if m == label
                    )
                    for label in labels
                ]
                for p in parameters
            ]
        )

    def at(
        self,
        ids: NDArray[np.int64],
        mode: NDArray[np.int64],
        temperature: NDArray[np.float64],
        covariates: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]
    ]:
        location = (
            self.beta[ids]
            + self.idle[ids, mode]
            + (self.weather[ids] * covariates).sum(axis=1)
        )
        a = np.maximum(self.tbal[ids] - temperature, 0) * np.exp(-location)
        r = self.rho[ids] * np.clip(1 + self.slope[ids] * (temperature - 7), 0.4, 1.3)
        over = a >= (1 - OVERLOAD_MARGIN) * r
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = -np.log(a)
            mr = np.where(
                over, np.inf, -np.log(np.where(over, 1.0, r - a)) + self.run[ids, mode]
            )
        return mi, mr, over, np.maximum((a - (1 - OVERLOAD_MARGIN) * r) / r, 0)


def simulate(
    request: HvacForecastRequest,
    fit: CycleFit,
    power: PowerFit,
    initial: InitialState,
    cfg: HvacModelConfig,
    capabilities: HvacCapabilities,
    outdoor_c: NDArray[np.float32],
    sun: NDArray[np.float64],
    modes: NDArray[np.str_],
    rng: np.random.Generator,
    *,
    supply_c: float | None = None,
    cancel: Event | None = None,
    deadline: float | None = None,
) -> SimulationResult:
    """Simulate event cohorts into minute samples, preserving commands and lockout.

    Args:
        request: Actual origin and future output horizon.
        fit: Full-data and bootstrap cycle fits.
        power: Meter-scope electrical model and observed low states.
        initial: Conditional first phase, including water/controller mixtures.
        cfg: Path count, local covariates and duration dispersion.
        capabilities: Manufacturer minimum spells and optional backup heater.
        outdoor_c: All-path minute outdoor trajectories.
        sun: Minute radiation in kW/m².
        modes: Effective mode per path/minute from schedule_modes.
        rng: One prediction-owned deterministic generator.
        supply_c: Observed current supply, held over the horizon; otherwise the
            fitted run mean is used and reported. No future supply is invented.
        cancel: Cooperative cancellation.
        deadline: Absolute monotonic deadline, defaulting to the request budget.

    Returns:
        Full immutable paths and diagnostics. Event times retain fractional
        minutes; output power/state samples use minute starts.

    Raises:
        ValueError: For mismatched trajectories or missing defrost power.
        HvacCancelled: On cancellation between event cohorts.
        HvacTimeout: When the prediction deadline is exhausted.
    """
    deadline = deadline if deadline is not None else monotonic() + request.deadline_s
    check_budget(cancel, deadline)
    start = pd.Timestamp(request.origin).floor("min").to_pydatetime()
    horizon = int(np.ceil((request.grid[-1].end - start).total_seconds() / 60))
    n = cfg.n_paths
    if (
        outdoor_c.shape != (n, horizon)
        or modes.shape != (n, horizon)
        or sun.shape != (horizon,)
        or not np.isfinite(outdoor_c).all()
        or not np.isfinite(sun).all()
    ):
        raise HvacError("simulation trajectories have invalid shape or values")
    if (
        fit.defrost_length_min > 0
        and max(*fit.defrost_rate, *fit.defrost_hourly_rate) > 0
        and power.defrost_w is None
    ):
        raise HvacError("defrost rate has no measured interruption power")
    actual_mode = request.last_mode or cfg.reference_mode
    uniform_mode = bool(np.all(modes == modes[0, 0]))
    if uniform_mode:
        labels = np.unique([modes[0, 0], actual_mode])
        constant_mode_id = int(np.searchsorted(labels, modes[0, 0]))
        mode = np.broadcast_to(
            np.full((n, 1), constant_mode_id, dtype=np.int64), (n, horizon)
        )
        next_mode = np.broadcast_to(
            np.full((n, 1), horizon, dtype=np.int64), (n, horizon)
        )
        is_off = np.isin(labels, list(cfg.off_modes))
    else:
        labels = np.union1d(np.unique(modes), [actual_mode])
        mode = np.asarray(np.searchsorted(labels, modes), dtype=np.int64)
        is_off = np.isin(labels, list(cfg.off_modes))
        # Next schedule boundary per path; no Python per-minute scan.
        boundary = np.concatenate(
            [mode[:, 1:] != mode[:, :-1], np.zeros((n, 1), bool)], axis=1
        )
        next_mode = np.minimum.accumulate(
            np.where(boundary, np.arange(horizon) + 1, horizon)[:, ::-1], axis=1
        )[:, ::-1]
    rates = _Rates(fit, cfg, labels)
    index = pd.date_range(start, periods=horizon, freq="min")
    local = index.tz_convert(cfg.timezone)
    angle = 2 * np.pi * (local.hour + local.minute / 60) / 24
    cov = np.column_stack([sun, np.sin(angle), np.cos(angle)])
    all_ids = np.arange(n, dtype=np.int64)
    current_mode = np.full(n, int(np.searchsorted(labels, actual_mode)), dtype=np.int64)
    state = initial.running.copy()
    state[is_off[current_mode]] = False
    elapsed = initial.elapsed_min.copy()
    began = -elapsed
    offset = (request.origin - start).total_seconds() / 60
    began += offset
    remaining = initial.remaining_min.copy()
    min_run = capabilities.min_run_minutes or 0.0
    min_off = capabilities.min_off_minutes or 0.0
    remaining = np.maximum(
        remaining, np.maximum(np.where(state, min_run, min_off) - elapsed, 0)
    )
    remaining[~state] = np.maximum(remaining[~state], initial.lockout_min)
    natural = offset + remaining
    natural[is_off[current_mode]] = np.inf
    mi, mr, overloaded, depth = rates.at(
        all_ids,
        current_mode,
        outdoor_c[:, 0].astype(float),
        cov[0],
    )
    natural[state & overloaded] = np.inf
    level = np.maximum(
        power.run_power(
            outdoor_c[:, 0],
            supply_c if supply_c is not None else power.supply_reference_c,
        )
        + rng.normal(0, power.residual_sd_w, n),
        0,
    )
    diagnostics: set[str] = set()
    if power.uses_supply:
        source = (
            "control_setpoint"
            if power.temperature_source == "control_setpoint_c"
            else "supply"
        )
        diagnostics.add(
            f"{source}_held_at_origin"
            if supply_c is not None
            else f"{source}_from_fit_mean"
        )
    pending_recovery = is_off[current_mode].copy()
    water = initial.water_draw.copy()
    water_start = offset + initial.remaining_min
    no_demand_idle = ~state & np.isposinf(mi) & ~is_off[current_mode] & ~water
    off_started = np.where(is_off[current_mode], -elapsed + offset, np.nan)
    idle_power = np.array(
        [power.idle_by_mode.get(str(m), power.idle_default_w) for m in labels]
    )
    watts = np.empty((n, horizon), dtype=np.float32)
    running = np.empty((n, horizon), dtype=bool)
    starts = np.zeros((n, horizon), dtype=bool)
    filled = np.zeros(n, dtype=np.int64)
    # Defrost is sampled once for each new run; intervals are wall-clock offsets.
    defrost_start: list[NDArray[np.float64]] = [np.empty(0) for _ in range(n)]
    defrost_on: list[NDArray[np.bool_]] = [np.empty(0, bool) for _ in range(n)]

    def draw_defrost(
        ids: NDArray[np.int64],
        at: NDArray[np.float64],
        active: NDArray[np.float64],
        temp: NDArray[np.float64],
    ) -> None:
        bins = np.where(temp < 3, 0, np.where(temp <= 7, 1, 2))
        counts = rng.poisson(np.asarray(fit.defrost_rate)[bins])
        # One short loop per run cohort member, never per simulated minute.
        for j, path in enumerate(ids):
            count = int(counts[j]) if np.isfinite(active[j]) else 0
            if count == 0:
                defrost_start[path] = np.empty(0)
                defrost_on[path] = np.empty(0, bool)
                continue
            positions = np.sort(rng.uniform(0, active[j], count))
            defrost_start[path] = (
                at[j] + positions + np.arange(count) * fit.defrost_length_min
            )
            defrost_on[path] = rng.random(count) < power.defrost_running_share
        natural[ids] += counts * fit.defrost_length_min * np.isfinite(active)

    def draw_overload_defrost(ids: NDArray[np.int64], at: float) -> None:
        # The approved observed hourly hazard runs on active-compressor time.
        # Existing interruptions straddling a boundary remain intact.
        for path in ids:
            temp = outdoor_c[path, min(int(at), horizon - 1)]
            bin_index = 0 if temp < 3 else 1 if temp <= 7 else 2
            rate = fit.defrost_hourly_rate[bin_index]
            if rate <= 0 or fit.defrost_length_min <= 0:
                continue
            keep = defrost_start[path] < at
            previous = defrost_start[path][keep]
            previous_on = defrost_on[path][keep]
            clock = max(
                at,
                float(previous[-1] + fit.defrost_length_min) if len(previous) else at,
            )
            end = min((int(at) // 60 + 1) * 60, horizon)
            added: list[float] = []
            while clock < end:
                clock += float(rng.exponential(60 / rate))
                if clock < end:
                    added.append(clock)
                clock += fit.defrost_length_min
            defrost_start[path] = np.concatenate([previous, added])
            defrost_on[path] = np.concatenate(
                [previous_on, rng.random(len(added)) < power.defrost_running_share]
            )

    # Partial current run has no invented past defrost; future interruptions only.
    current_run = all_ids[state & np.isfinite(remaining)]
    draw_defrost(
        current_run,
        np.full(len(current_run), offset),
        remaining[current_run],
        outdoor_c[current_run, 0].astype(float),
    )
    draw_overload_defrost(all_ids[state & overloaded], offset)
    natural_index = np.full(n, horizon, dtype=np.int64)
    finite = np.isfinite(natural) & (natural < horizon)
    natural_index[finite] = np.ceil(natural[finite]).astype(np.int64)
    while np.any(filled < horizon):
        check_budget(cancel, deadline)
        position = np.minimum(filled, horizon - 1)
        mode_here = current_mode
        # Overload and an infinite zero-demand idle need hourly regime checks.
        hour = ((filled // 60) + 1) * 60
        hourly = np.where((state & overloaded) | no_demand_idle, hour, horizon)
        if uniform_mode:
            event = np.minimum(natural_index, hourly)
            # A uniform future mode can differ from the actual origin mode.
            # That common first boundary is still processed at minute zero.
            if current_mode[0] != constant_mode_id:
                event = np.minimum(event, filled)
        else:
            event = np.minimum.reduce(
                [
                    natural_index,
                    next_mode[all_ids, position],
                    hourly,
                    np.where(current_mode != mode[all_ids, position], filled, horizon),
                ]
            )
        event = np.maximum(event, filled)
        active = filled < horizon
        earliest = int(event[active].min())
        ids = all_ids[active & (event == earliest)]
        backup_possible = (
            capabilities.backup_heater_kw is not None
            and capabilities.backup_heater_max_outdoor_c is not None
            and np.any(state[ids] & overloaded[ids])
        )
        if not backup_possible:
            # Each event-bounded path segment has one mode and run level. Write
            # its contiguous slice, then at most three ramp samples and recorded
            # defrost slices; there is no Python loop over the minute horizon.
            for path in ids:
                left = int(filled[path])
                if is_off[current_mode[path]]:
                    watts[path, left:earliest] = power.off_w
                    running[path, left:earliest] = False
                elif not state[path]:
                    watts[path, left:earliest] = idle_power[current_mode[path]]
                    running[path, left:earliest] = False
                else:
                    watts[path, left:earliest] = level[path]
                    running[path, left:earliest] = True
                    ramp_end = min(earliest, int(np.ceil(began[path] + 3)))
                    for minute in range(left, ramp_end):
                        age = minute - began[path]
                        if 0 <= age < 3:
                            watts[path, minute] = (
                                level[path] * power.start_ramp[int(age)]
                            )
                    for d, on_defrost in zip(
                        defrost_start[path], defrost_on[path], strict=True
                    ):
                        first = max(left, int(np.ceil(d)))
                        last = min(earliest, int(np.ceil(d + fit.defrost_length_min)))
                        if first < last:
                            watts[path, first:last] = power.defrost_w
                            running[path, first:last] = on_defrost
        else:
            # Equal-event cohort can have different previous event positions.
            left = int(filled[ids].min())
            t = np.arange(left, earliest)
            mask = t[None, :] >= filled[ids, None]
            # A cohort ends at its next mode boundary, so every written minute has
            # its current mode. Masked prefixes need no historical mode expansion.
            m = current_mode[ids, None]
            values = np.broadcast_to(level[ids, None], mask.shape).copy()
            if np.any(state[ids] & (filled[ids] - began[ids] < 3)):
                age = t[None, :] - began[ids, None]
                ramp = np.clip(np.floor(age), 0, 2).astype(int)
                values *= np.where(
                    (age >= 0) & (age < 3), np.asarray(power.start_ramp)[ramp], 1.0
                )
            values = np.where(state[ids, None], values, idle_power[m])
            values = np.where(is_off[m], power.off_w, values)
            on = np.broadcast_to(state[ids, None], mask.shape).copy()
            if (
                capabilities.backup_heater_kw is not None
                and capabilities.backup_heater_max_outdoor_c is not None
            ):
                backup = (state[ids] & overloaded[ids])[:, None] & (
                    outdoor_c[ids[:, None], t]
                    <= capabilities.backup_heater_max_outdoor_c
                )
                values += (
                    backup
                    * (
                        capabilities.backup_heater_kw * 1000 * np.clip(depth[ids], 0, 1)
                    )[:, None]
                )
                if backup.any():
                    diagnostics.add("backup_heater_assumed")
            for j, path in enumerate(ids):
                for d, on_defrost in zip(
                    defrost_start[path], defrost_on[path], strict=True
                ):
                    use = (t >= d) & (t < d + fit.defrost_length_min) & state[path]
                    values[j, use] = power.defrost_w
                    on[j, use] = on_defrost
            rows, cols = np.broadcast_arrays(ids[:, None], t[None, :])
            watts[rows[mask], cols[mask]] = values[mask]
            running[rows[mask], cols[mask]] = on[mask]
        previous = mode_here[ids]
        filled[ids] = earliest
        if earliest >= horizon:
            continue
        new = mode[ids, earliest]
        current_mode[ids] = new
        changed = new != previous
        old_state = state[ids].copy()
        old_over = overloaded[ids].copy()
        temp = outdoor_c[ids, earliest].astype(float)
        mi_new, mr_new, over_new, depth_new = rates.at(ids, new, temp, cov[earliest])
        if changed.any():
            mi_old, mr_old, _, _ = rates.at(ids, previous, temp, cov[earliest])
            # A schedule command takes precedence over a simultaneous natural event.
            into_off = changed & is_off[new]
            out_off = changed & is_off[previous] & ~is_off[new]
            state[ids[into_off]] = False
            off_started[ids[into_off]] = earliest
            pending_recovery[ids[into_off]] = True
            natural[ids[into_off]] = np.inf
            began[ids[into_off & old_state]] = earliest
            for off_index in np.flatnonzero(into_off):
                defrost_start[ids[off_index]] = np.empty(0)
                defrost_on[ids[off_index]] = np.empty(0, bool)
            if out_off.any():
                out_ids = ids[out_off]
                age_off = np.maximum(earliest - off_started[out_ids], 0)
                natural[out_ids] = earliest + conditional_lognormal(
                    mi_new[out_off], rates.si[out_ids], age_off, rng
                )
                water_ids = out_ids[water[out_ids]]
                natural[water_ids] = np.maximum(water_start[water_ids], earliest)
                # Keep the physical off clock, including an idle before the command.
            rescale = changed & ~into_off & ~out_off & ~water[ids]
            for on_state in (False, True):
                use = rescale & (old_state == on_state)
                old_mu, new_mu = (mr_old, mr_new) if on_state else (mi_old, mi_new)
                finite_pair = use & np.isfinite(old_mu) & np.isfinite(new_mu)
                natural[ids[finite_pair]] = earliest + np.maximum(
                    natural[ids[finite_pair]] - earliest, 0
                ) * np.exp(new_mu[finite_pair] - old_mu[finite_pair])
            # Learned outcomes replace generic scaling, including OFF release.
            for effect in fit.switch_effects:
                use = (
                    changed
                    & ~water[ids]
                    & (labels[previous] == effect.from_mode)
                    & (labels[new] == effect.to_mode)
                    & ~is_off[new]
                    & (
                        ("run" if effect.state == "run" else "idle")
                        == np.where(state[ids], "run", "idle")
                    )
                )
                if use.any():
                    chosen = ids[use]
                    natural[chosen] = earliest + rng.lognormal(
                        effect.mu, effect.sigma * cfg.dispersion_inflation, len(chosen)
                    )
                    diagnostics.add(
                        f"switch_effect_learned:{effect.from_mode}->{effect.to_mode}:{effect.state}"
                    )
        # Overload is a regime, not a very small run rate.
        entering = state[ids] & over_new & ~is_off[new]
        natural[ids[entering]] = np.inf
        leaving = state[ids] & old_over & ~over_new & ~is_off[new]
        if leaving.any():
            chosen = ids[leaving]
            natural[chosen] = earliest + conditional_lognormal(
                mr_new[leaving], rates.sr[chosen], earliest - began[chosen], rng
            )
        overloaded[ids] = over_new
        depth[ids] = depth_new
        flip = (natural_index[ids] <= earliest) & ~changed & ~is_off[new] & ~entering
        # An hourly overload exit resamples a residual instead of stopping immediately.
        flip &= ~leaving
        if flip.any():
            chosen = ids[flip]
            state[chosen] = ~state[chosen]
            began[chosen] = natural[chosen]
            water[chosen] = False
            on_new = state[chosen]
            mu = np.where(on_new, mr_new[flip], mi_new[flip])
            sigma = np.where(on_new, rates.sr[chosen], rates.si[chosen])
            duration = np.exp(
                np.minimum(mu + sigma * rng.normal(size=len(chosen)), 700)
            )
            duration[np.isposinf(mu)] = np.inf
            duration[on_new & pending_recovery[chosen]] *= rates.recovery[
                chosen[on_new & pending_recovery[chosen]]
            ]
            duration = np.maximum(duration, np.where(on_new, min_run, min_off))
            natural[chosen] = began[chosen] + np.maximum(duration, 1e-6)
            newly_running = chosen[on_new]
            starts[newly_running, earliest] = True
            pending_recovery[newly_running] = False
            level[newly_running] = np.maximum(
                power.run_power(
                    outdoor_c[newly_running, earliest],
                    supply_c if supply_c is not None else power.supply_reference_c,
                )
                + rng.normal(0, power.residual_sd_w, len(newly_running)),
                0,
            )
            for path in chosen[~on_new]:
                defrost_start[path] = np.empty(0)
                defrost_on[path] = np.empty(0, bool)
            draw_defrost(
                newly_running,
                began[newly_running],
                duration[on_new],
                outdoor_c[newly_running, earliest].astype(float),
            )
        resumed = no_demand_idle[ids] & ~state[ids] & ~is_off[new] & np.isfinite(mi_new)
        if resumed.any():
            chosen = ids[resumed]
            natural[chosen] = earliest + conditional_lognormal(
                mi_new[resumed], rates.si[chosen], np.zeros(len(chosen)), rng
            )
            diagnostics.add("zero_demand_idle_resumed")
        no_demand_idle[ids] = (
            ~state[ids] & ~is_off[new] & np.isposinf(mi_new) & ~water[ids]
        )
        natural[ids[no_demand_idle[ids]]] = np.inf
        # Minimum spells survive mode rescaling, but OFF commands always win.
        eligible = ~is_off[new]
        lower = began[ids] + np.where(state[ids], min_run, min_off)
        lower = np.where(
            state[ids], lower, np.maximum(lower, offset + initial.lockout_min)
        )
        natural[ids[eligible]] = np.maximum(natural[ids[eligible]], lower[eligible])
        # A zero-residual learned/conditional event is processed at the next sample.
        natural[ids[eligible]] = np.maximum(natural[ids[eligible]], earliest + 1e-6)
        draw_overload_defrost(ids[state[ids] & overloaded[ids]], float(earliest))
        # Only this cohort changed its natural clock. Other paths retain their
        # cached sample boundary, including censored/infinite clocks.
        natural_index[ids] = horizon
        finite = np.isfinite(natural[ids]) & (natural[ids] < horizon)
        natural_index[ids[finite]] = np.ceil(natural[ids[finite]]).astype(np.int64)
    return SimulationResult(
        MinutePaths(start, watts, running, starts), tuple(sorted(diagnostics))
    )
