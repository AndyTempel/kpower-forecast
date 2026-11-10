"""Minute-stepped fixed-speed cycle simulation over the full path ensemble.

Each path alternates run and idle spells. A new spell's length is drawn from the
fitted log-normal at the path's current outdoor temperature, radiation, mode and
(for idle) the length of the run that just ended. A mode change resamples the
remaining time conditional on the elapsed time; an off mode stops the
compressor until it is released. No network access and no model installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._stats import check_budget, conditional_lognormal
from .cycle_model import CycleFit
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


@dataclass(frozen=True)
class InitialState:
    """Observed phase at the origin; ``running is None`` when it is unknown."""

    running: bool | None
    elapsed_min: float = 0.0
    prev_run_min: float | None = None
    lockout_min: float = 0.0


def initial_state(
    recent: pd.DataFrame, origin: datetime, cfg: HvacModelConfig
) -> InitialState:
    """Read the current spell and its age from cleaned recent minutes.

    Args:
        recent: Regular minute frame with cleaned nullable ``running`` (and an
            optional ``lockout_remaining_s``), already sliced to availability.
        origin: Actual forecast origin.
        cfg: Freshness and gap tolerances.

    Returns:
        The phase, or an unknown phase when the newest state is stale.
    """
    if "running" not in recent:
        return InitialState(None)
    state = recent.running.dropna()
    if state.empty:
        return InitialState(None)
    newest = state.index[-1].to_pydatetime() + timedelta(minutes=1)
    if (origin - newest).total_seconds() / 60 > cfg.state_max_age_min:
        return InitialState(None)
    values = state.to_numpy()
    stamps = state.index
    tolerance = pd.Timedelta(minutes=cfg.minute_gap_tolerance_min)
    on = bool(values[-1])
    i = len(values) - 1
    while (
        i > 0 and values[i - 1] == values[-1] and stamps[i] - stamps[i - 1] <= tolerance
    ):
        i -= 1
    elapsed = max((origin - stamps[i].to_pydatetime()).total_seconds() / 60, 0.0)
    prev_run = None
    if (
        not on
        and i > 0
        and values[i - 1] == 1
        and stamps[i] - stamps[i - 1] <= tolerance
    ):
        j = i - 1
        while j > 0 and values[j - 1] == 1 and stamps[j] - stamps[j - 1] <= tolerance:
            j -= 1
        prev_run = (stamps[i - 1] - stamps[j]).total_seconds() / 60 + 1
    lockout = 0.0
    if "lockout_remaining_s" in recent:
        countdown = recent.lockout_remaining_s.dropna()
        if len(countdown):
            age = (origin - countdown.index[-1].to_pydatetime()).total_seconds() - 60
            lockout = max(float(countdown.iloc[-1]) - max(age, 0.0), 0.0) / 60
    return InitialState(on, elapsed, prev_run, lockout)


@dataclass(frozen=True)
class ModeGrid:
    """Effective mode per path and minute, as small integer codes into labels."""

    labels: tuple[str, ...]
    codes: NDArray[np.int16]

    @classmethod
    def from_names(cls, names: NDArray[np.str_]) -> ModeGrid:
        """Encode a string mode array (convenience for tests and tools)."""
        unique, inverse = np.unique(names, return_inverse=True)
        return cls(
            tuple(str(v) for v in unique),
            inverse.reshape(names.shape).astype(np.int16),
        )

    def names(self) -> NDArray[np.str_]:
        """Decode to mode names (allocates; not used on the prediction path)."""
        return np.asarray(self.labels)[self.codes]


def schedule_modes(
    request: HvacForecastRequest,
    cfg: HvacModelConfig,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    *,
    comfort_off: tuple[float, float] | None = None,
) -> tuple[ModeGrid, tuple[str, ...]]:
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
    ordered = tuple(sorted(labels))
    code = {label: np.int16(i) for i, label in enumerate(ordered)}
    modes = np.full(
        (cfg.n_paths, len(index)),
        code[request.last_mode or cfg.reference_mode],
        dtype=np.int16,
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
            modes[:, start:end] = code[span.mode]
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
            at[None, :] < release[:, None], code[span.mode], code[span.release_mode]
        )
    return ModeGrid(ordered, modes), tuple(diagnostics)


class _Locations:
    """Duration-model coefficients packed per mode code for vector evaluation."""

    def __init__(self, fit: CycleFit, labels: list[str]) -> None:
        width = len(fit.modes)
        self.idle = np.asarray(fit.idle_beta)
        self.run = np.asarray(fit.run_beta)
        self.idle_mode = np.zeros(len(labels))
        self.run_mode = np.zeros(len(labels))
        for code, label in enumerate(labels):
            if label in fit.modes:
                k = fit.modes.index(label)
                self.idle_mode[code] = self.idle[3 + k]
                self.run_mode[code] = self.run[3 + k]
        self.prev = self.idle[3 + width]

    def __call__(
        self,
        running: NDArray[np.bool_],
        outdoor: NDArray[np.floating],
        sun: float,
        code: NDArray[np.integer],
        prev_run: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        mu_run = (
            self.run[0]
            + self.run[1] * outdoor
            + self.run[2] * sun
            + self.run_mode[code]
        )
        mu_idle = (
            self.idle[0]
            + self.idle[1] * outdoor
            + self.idle[2] * sun
            + self.idle_mode[code]
            + self.prev * np.log(np.maximum(prev_run, 1.0))
        )
        return np.asarray(np.where(running, mu_run, mu_idle))


def simulate(
    request: HvacForecastRequest,
    fit: CycleFit,
    power: PowerFit,
    initial: InitialState,
    cfg: HvacModelConfig,
    capabilities: HvacCapabilities,
    outdoor_c: NDArray[np.float32],
    sun: NDArray[np.float64],
    modes: ModeGrid,
    rng: np.random.Generator,
    *,
    cancel: Event | None = None,
    deadline: float | None = None,
) -> SimulationResult:
    """Simulate all paths minute by minute.

    Args:
        request: Actual origin and output horizon.
        fit: Duration models.
        power: Electrical model.
        initial: Observed phase (or unknown) at the origin.
        cfg: Path count and off modes.
        capabilities: Manufacturer minimum run/off times.
        outdoor_c: Per-path minute outdoor temperatures ``[path, minute]``.
        sun: Minute radiation in kW/m².
        modes: Effective mode per path/minute from :func:`schedule_modes`.
        rng: Prediction-owned deterministic generator.
        cancel: Cooperative cancellation.
        deadline: Absolute monotonic deadline.

    Returns:
        Full immutable paths (power, running, starts) and diagnostics.

    Raises:
        ValueError: For mismatched trajectory shapes.
    """
    check_budget(cancel, deadline)
    start = pd.Timestamp(request.origin).floor("min").to_pydatetime()
    horizon = int(np.ceil((request.grid[-1].end - start).total_seconds() / 60))
    n = cfg.n_paths
    if (
        outdoor_c.shape != (n, horizon)
        or modes.codes.shape != (n, horizon)
        or sun.shape != (horizon,)
        or not np.isfinite(outdoor_c).all()
        or not np.isfinite(sun).all()
    ):
        raise HvacError("simulation trajectories do not match the path horizon")
    labels = list(modes.labels)
    codes = modes.codes
    is_off_code = np.array([label in cfg.off_modes for label in labels])
    locate = _Locations(fit, labels)
    min_run = float(capabilities.min_run_minutes or 0.0)
    min_off = float(capabilities.min_off_minutes or 0.0)
    temperature = outdoor_c  # float32; per-path slices are cast on use
    diagnostics: set[str] = set()
    paths = np.arange(n)
    # Offset of the actual origin within the first minute sample.
    lead = (request.origin - start).total_seconds() / 60
    prev_run = np.full(n, initial.prev_run_min or fit.typical_run_min)
    if initial.running is None:
        diagnostics.add("stationary_initial_state")
        mu_i = locate(
            np.zeros(n, bool), temperature[:, 0], sun[0], codes[:, 0], prev_run
        )
        mu_r = locate(
            np.ones(n, bool), temperature[:, 0], sun[0], codes[:, 0], prev_run
        )
        mean_i = np.exp(mu_i + fit.idle_sigma**2 / 2)
        mean_r = np.exp(mu_r + fit.run_sigma**2 / 2)
        running = rng.random(n) < mean_r / (mean_r + mean_i)
        sigma = np.where(running, fit.run_sigma, fit.idle_sigma)
        mu = np.where(running, mu_r, mu_i)
        # Length-biased draw of the current spell, then a uniform position in it.
        length = np.exp(np.minimum(mu + sigma**2 + sigma * rng.normal(size=n), 50))
        elapsed = rng.random(n) * length
        remaining = length - elapsed
    else:
        running = np.full(n, initial.running)
        elapsed = np.full(n, initial.elapsed_min)
        mu = locate(running, temperature[:, 0], sun[0], codes[:, 0], prev_run)
        sigma = np.where(running, fit.run_sigma, fit.idle_sigma)
        remaining = conditional_lognormal(mu, sigma, elapsed, rng)
        remaining = np.maximum(
            remaining,
            np.where(running, min_run, max(min_off, initial.lockout_min)) - elapsed,
        )
    remaining = np.maximum(remaining - lead, 0.0)
    off = is_off_code[codes[:, 0]]
    running[off] = False
    remaining[off] = np.inf
    initial_minute = np.where(running, elapsed, 0.0)
    state = np.empty((n, horizon), bool)
    change = np.zeros(horizon, bool)
    change[1:] = (codes[:, 1:] != codes[:, :-1]).any(axis=0)
    for j in range(horizon):
        if j % 720 == 0:
            check_budget(cancel, deadline)
        if change[j]:
            ids = paths[codes[:, j] != codes[:, j - 1]]
            now_off = is_off_code[codes[ids, j]]
            stopped = ids[now_off]
            running[stopped] = False
            remaining[stopped] = np.inf
            elapsed[stopped] = 0.0
            resumed = ids[~now_off]
            if len(resumed):
                mu = locate(
                    running[resumed],
                    temperature[resumed, j],
                    sun[j],
                    codes[resumed, j],
                    prev_run[resumed],
                )
                sigma = np.where(running[resumed], fit.run_sigma, fit.idle_sigma)
                remaining[resumed] = conditional_lognormal(
                    mu, sigma, elapsed[resumed], rng
                )
        state[:, j] = running
        elapsed += 1.0
        remaining -= 1.0
        switch = remaining <= 0
        if switch.any():
            ids = paths[switch]
            ended_run = running[ids]
            prev_run[ids[ended_run]] = elapsed[ids[ended_run]]
            running[ids] = ~running[ids]
            elapsed[ids] = 0.0
            mu = locate(
                running[ids], temperature[ids, j], sun[j], codes[ids, j], prev_run[ids]
            )
            sigma = np.where(running[ids], fit.run_sigma, fit.idle_sigma)
            drawn = np.exp(np.minimum(mu + sigma * rng.normal(size=len(ids)), 50))
            remaining[ids] = np.maximum(drawn, np.where(running[ids], min_run, min_off))
    # Minute within the current run and power, vectorized after the event loop
    # in column blocks so temporaries stay small on the controller.
    starts = np.zeros((n, horizon), bool)
    starts[:, 1:] = state[:, 1:] & ~state[:, :-1]
    watts = np.empty((n, horizon), np.float32)
    begin = np.where(state[:, 0], 0, -1).astype(np.int32)
    offset = initial_minute * state[:, 0]
    block = 720
    for a in range(0, horizon, block):
        b = min(a + block, horizon)
        column = np.arange(a, b, dtype=np.int32)
        marks = np.where(starts[:, a:b], column, -1).astype(np.int32)
        marks[:, 0] = np.maximum(marks[:, 0], begin)
        marks = np.maximum.accumulate(marks, axis=1)
        begin = marks[:, -1]
        minute = (column - marks) + np.where(marks == 0, offset[:, None], 0.0)
        level = power.run_power(minute, temperature[:, a:b])
        low = np.where(is_off_code[codes[:, a:b]], power.off_w, power.idle_w)
        watts[:, a:b] = np.where(state[:, a:b], level, low)
    return SimulationResult(
        MinutePaths(start, watts, state, starts),
        tuple(sorted(diagnostics)),
    )
