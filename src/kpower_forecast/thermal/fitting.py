"""Shared bounded RC fit search for runtime and offline diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass(frozen=True)
class FitCandidate:
    """One diagnostic optimum; rejected coefficients remain diagnostic only."""

    tau_hours: float
    gain_c_per_kw: float
    offset_c: float
    loss_c2: float
    condition_number: float
    boundary: str | None
    profile_relative_range: float

    @property
    def parameters(self) -> tuple[float, float, float]:
        """Return the coefficients in the model's canonical order."""
        return self.tau_hours, self.gain_c_per_kw, self.offset_c


def fit_candidate(
    starts: np.ndarray,
    ends: np.ndarray,
    outdoors: np.ndarray,
    power_kw: np.ndarray,
    hours: np.ndarray,
    *,
    min_tau: float,
    max_tau: float,
    grid_points: int,
    weights: np.ndarray | None = None,
) -> FitCandidate | None:
    """Search the existing RC grid, optionally with offline confidence weights.

    Args:
        starts: Genuine indoor start readings.
        ends: Genuine indoor end readings.
        outdoors: Covered mean outdoor forcing.
        power_kw: Attributable mean HVAC electric input.
        hours: Exact elapsed observation times.
        min_tau: Lower physical time-constant limit.
        max_tau: Upper physical time-constant limit.
        grid_points: Number of logarithmically spaced candidates.
        weights: Optional finite positive row weights; runtime omits them.

    Returns:
        Best unconstrained coefficients on the bounded grid, or None if singular.

    Raises:
        ValueError: If weights are malformed.
    """
    if weights is not None and (
        weights.shape != starts.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("fit weights must be finite, positive and aligned")
    best: tuple[float, float, float, float, float] | None = None
    losses: list[float] = []
    for tau in np.geomspace(min_tau, max_tau, grid_points):
        decay = -np.expm1(-hours / tau)
        target = ends - (1 - decay) * starts - decay * outdoors
        design = np.column_stack((decay * power_kw, decay))
        solve_design, solve_target = design, target
        if weights is not None:
            root = np.sqrt(weights)
            solve_design, solve_target = design * root[:, None], target * root
        if np.linalg.matrix_rank(solve_design) < 2:
            continue
        gain, offset = np.linalg.lstsq(solve_design, solve_target, rcond=None)[0]
        residual = target - design @ np.array([gain, offset])
        loss = float(
            np.mean(residual**2)
            if weights is None
            else np.average(residual**2, weights=weights)
        )
        if not np.isfinite(loss):
            continue
        losses.append(loss)
        if best is None or loss < best[3]:
            best = (
                float(tau),
                float(gain),
                float(offset),
                loss,
                float(np.linalg.cond(solve_design)),
            )
    if best is None:
        return None
    boundary = (
        "lower" if best[0] == min_tau else "upper" if best[0] == max_tau else None
    )
    return FitCandidate(
        *best,
        boundary=boundary,
        profile_relative_range=(max(losses) - min(losses)) / max(max(losses), 1e-12),
    )


@dataclass(frozen=True)
class SimulationWindows:
    """Flattened multi-step training windows over real indoor readings.

    Every segment ends at a genuine reading. A bridged gap segment carries
    filled *inputs* only; its end target is still the real reading that
    resumed the chain, and no indoor temperature is ever interpolated.
    """

    window_ids: np.ndarray
    initial_c: np.ndarray
    hours: np.ndarray
    outdoor_c: np.ndarray
    power_kw: np.ndarray
    observed_c: np.ndarray
    reading_ids: np.ndarray
    bridged_hours: float

    @property
    def window_count(self) -> int:
        """Return the number of simulated windows."""
        return int(self.initial_c.size)

    @property
    def reading_count(self) -> int:
        """Return distinct real readings scored, ignoring window overlap."""
        return int(np.unique(self.reading_ids).size)


@dataclass(frozen=True)
class ChainSegment:
    """One constant-input interval that ends at a real indoor reading."""

    start_hours: float
    hours: float
    start_c: float
    end_c: float
    outdoor_c: float
    power_kw: float
    bridged: bool


def build_chains(
    starts_h: np.ndarray,
    ends_h: np.ndarray,
    start_c: np.ndarray,
    end_c: np.ndarray,
    outdoor_c: np.ndarray,
    power_kw: np.ndarray,
    *,
    max_bridge_hours: float,
) -> list[list[ChainSegment]]:
    """Link chronological transitions into contiguous simulation chains.

    A short gap between two covered transitions (typically one rejected for
    HVAC coverage) is bridged with the mean of the adjacent inputs. Longer
    gaps, or any overlap, start a new chain.
    """
    chains: list[list[ChainSegment]] = []
    current: list[ChainSegment] = []
    for index in range(starts_h.size):
        if current:
            previous_end = current[-1].start_hours + current[-1].hours
            gap = float(starts_h[index] - previous_end)
            if gap < -1e-9 or gap > max_bridge_hours:
                chains.append(current)
                current = []
            elif gap > 1e-9:
                current.append(
                    ChainSegment(
                        start_hours=previous_end,
                        hours=gap,
                        start_c=current[-1].end_c,
                        end_c=float(start_c[index]),
                        outdoor_c=float((current[-1].outdoor_c + outdoor_c[index]) / 2),
                        power_kw=float((current[-1].power_kw + power_kw[index]) / 2),
                        bridged=True,
                    )
                )
        current.append(
            ChainSegment(
                start_hours=float(starts_h[index]),
                hours=float(ends_h[index] - starts_h[index]),
                start_c=float(start_c[index]),
                end_c=float(end_c[index]),
                outdoor_c=float(outdoor_c[index]),
                power_kw=float(power_kw[index]),
                bridged=False,
            )
        )
    if current:
        chains.append(current)
    return chains


def build_windows(
    chains: list[list[ChainSegment]],
    *,
    min_hours: float,
    max_hours: float,
    stride_hours: float,
    max_bridge_fraction: float,
) -> SimulationWindows:
    """Cut overlapping 6-24 h style windows that start at real readings."""
    window_ids: list[int] = []
    initial: list[float] = []
    hours: list[float] = []
    outdoor: list[float] = []
    power: list[float] = []
    observed: list[float] = []
    reading_ids: list[int] = []
    bridged_total = 0.0
    reading_base = 0
    for chain in chains:
        starts = np.array([segment.start_hours for segment in chain])
        next_start: float | None = None
        for first, segment in enumerate(chain):
            if next_start is not None and segment.start_hours < next_start - 1e-9:
                continue
            last = first
            span = 0.0
            bridged = 0.0
            selected: list[ChainSegment] = []
            while last < len(chain) and span + chain[last].hours <= max_hours + 1e-9:
                selected.append(chain[last])
                span += chain[last].hours
                bridged += chain[last].hours if chain[last].bridged else 0.0
                last += 1
            if span < min_hours or bridged > max_bridge_fraction * span + 1e-9:
                continue
            window = len(initial)
            initial.append(segment.start_c)
            for offset, item in enumerate(selected):
                window_ids.append(window)
                hours.append(item.hours)
                outdoor.append(item.outdoor_c)
                power.append(item.power_kw)
                observed.append(item.end_c)
                reading_ids.append(reading_base + first + offset)
            bridged_total += bridged
            next_start = float(starts[first]) + stride_hours
        reading_base += len(chain)
    return SimulationWindows(
        window_ids=np.array(window_ids, dtype=int),
        initial_c=np.array(initial, dtype=float),
        hours=np.array(hours, dtype=float),
        outdoor_c=np.array(outdoor, dtype=float),
        power_kw=np.array(power, dtype=float),
        observed_c=np.array(observed, dtype=float),
        reading_ids=np.array(reading_ids, dtype=int),
        bridged_hours=bridged_total,
    )


def simulate_basis(
    windows: SimulationWindows, tau_hours: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the free, per-kW and per-offset responses at every reading.

    The first-order response is linear in gain and offset, so the simulated
    state at each reading is ``free + gain * per_kw + offset * per_offset``.
    The recurrence is solved in closed form with window-local cumulative
    decay; windows are at most a few dozen time constants long, so the
    exponentials stay well inside double precision.
    """
    retain = np.exp(-windows.hours / tau_hours)
    log_retain = -windows.hours / tau_hours
    cumulative = np.cumsum(log_retain)
    starts = np.r_[0, np.flatnonzero(np.diff(windows.window_ids)) + 1]
    before = np.repeat(
        cumulative[starts] - log_retain[starts],
        np.diff(np.r_[starts, windows.window_ids.size]),
    )
    local = cumulative - before
    scale = np.exp(-local)

    def response(drive: np.ndarray) -> np.ndarray:
        terms = (1 - retain) * drive * scale
        running = np.cumsum(terms)
        offsets = np.repeat(
            running[starts] - terms[starts],
            np.diff(np.r_[starts, windows.window_ids.size]),
        )
        return cast(np.ndarray, np.exp(local) * (running - offsets))

    free = np.exp(local) * windows.initial_c[windows.window_ids] + response(
        windows.outdoor_c
    )
    return free, response(windows.power_kw), response(np.ones_like(windows.hours))


@dataclass(frozen=True)
class OutputErrorFit:
    """Multi-step simulation fit with identifiability evidence."""

    tau_hours: float
    gain_c_per_kw: float
    offset_c: float
    mse_c2: float
    flags: tuple[str, ...]
    profile_relative_range: float
    window_count: int
    reading_count: int

    @property
    def parameters(self) -> tuple[float, float, float]:
        """Return the coefficients in the model's canonical order."""
        return self.tau_hours, self.gain_c_per_kw, self.offset_c


def _bounded_least_squares(
    target: np.ndarray,
    per_kw: np.ndarray,
    per_offset: np.ndarray,
    *,
    gain_bounds: tuple[float, float],
    offset_bound: float,
) -> tuple[float, float, float, tuple[str, ...]] | None:
    """Solve two coefficients by LS, falling back to the best box edge."""
    design = np.column_stack((per_kw, per_offset))
    if np.linalg.matrix_rank(design) == 2:
        gain, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        if gain_bounds[0] <= gain <= gain_bounds[1] and abs(offset) <= offset_bound:
            residual = target - design @ np.array([gain, offset])
            return float(gain), float(offset), float(np.mean(residual**2)), ()

    def solve_one(free: np.ndarray, target_rest: np.ndarray) -> float:
        energy = float(free @ free)
        return 0.0 if energy <= 1e-12 else float(free @ target_rest) / energy

    candidates: list[tuple[float, float, float, tuple[str, ...]]] = []
    for gain in gain_bounds:
        offset = float(
            np.clip(
                solve_one(per_offset, target - gain * per_kw),
                -offset_bound,
                offset_bound,
            )
        )
        residual = target - gain * per_kw - offset * per_offset
        candidates.append(
            (gain, offset, float(np.mean(residual**2)), ("gain_constrained",))
        )
    for offset in (-offset_bound, offset_bound):
        gain = float(
            np.clip(solve_one(per_kw, target - offset * per_offset), *gain_bounds)
        )
        residual = target - gain * per_kw - offset * per_offset
        candidates.append(
            (gain, offset, float(np.mean(residual**2)), ("offset_constrained",))
        )
    best = min(candidates, key=lambda item: item[2])
    return best if np.isfinite(best[2]) else None


def fit_output_error(
    windows: SimulationWindows,
    *,
    min_tau: float,
    max_tau: float,
    grid_points: int,
    tau_prior_median: float,
    tau_prior_log_sigma: float,
    gain_bounds: tuple[float, float],
    offset_bound: float,
    min_profile_relative_range: float,
) -> OutputErrorFit | None:
    """Fit tau by a regularised grid search on simulated multi-step error.

    The objective is the profile negative log posterior: Gaussian residuals
    with unknown variance over distinct real readings, plus a weak log-normal
    prior on tau. Gain and offset are bounded least squares on the simulated
    response; hitting a bound or the tau grid edge is reported as a flag
    instead of rejecting the fit.
    """
    if windows.window_count == 0:
        return None
    readings = max(windows.reading_count, 1)
    best: tuple[float, float, float, float, float, tuple[str, ...]] | None = None
    losses: list[float] = []
    for tau in np.geomspace(min_tau, max_tau, grid_points):
        free, per_kw, per_offset = simulate_basis(windows, float(tau))
        solved = _bounded_least_squares(
            windows.observed_c - free,
            per_kw,
            per_offset,
            gain_bounds=gain_bounds,
            offset_bound=offset_bound,
        )
        if solved is None:
            continue
        gain, offset, mse, flags = solved
        losses.append(mse)
        objective = 0.5 * readings * math.log(max(mse, 1e-12)) + (
            math.log(tau / tau_prior_median) ** 2
        ) / (2 * tau_prior_log_sigma**2)
        if best is None or objective < best[4]:
            best = (float(tau), gain, offset, mse, objective, flags)
    if best is None:
        return None
    found = list(best[5])
    if best[0] in (min_tau, max_tau):
        found.append("tau_boundary")
    profile = (max(losses) - min(losses)) / max(max(losses), 1e-12)
    if profile < min_profile_relative_range:
        found.append("flat_profile")
    return OutputErrorFit(
        tau_hours=best[0],
        gain_c_per_kw=best[1],
        offset_c=best[2],
        mse_c2=best[3],
        flags=tuple(found),
        profile_relative_range=profile,
        window_count=windows.window_count,
        reading_count=windows.reading_count,
    )
