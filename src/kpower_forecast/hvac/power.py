"""Measured electrical power while running, idle and off.

A fixed-speed compressor's electrical input rises through a run as the supply
water warms (observed: about 4.0 kW at start to 4.8 kW after ~15 minutes, with
a partial first minute). A flat level underestimates quarter-hour peaks, and a
straight line overstates long runs, so run power is the measured profile
by minute-in-run (mean) plus a linear outdoor-temperature term:

    P(k, T) = profile[min(k, K)] + c * (T - 7)

``profile`` is the mean over runs with known starts (including each run's
partial stop minute, so energy is preserved), up to the last minute observed in
enough runs; later minutes hold the last value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._stats import huber_regression
from .cycles import modes_at
from .types import HvacError, HvacModelConfig, MeteredScope

MIN_RUN_MINUTES_FOR_FIT: int = 30
MIN_RUNS_PER_PROFILE_MINUTE: int = 15
MAX_PROFILE_MINUTES: int = 90


@dataclass(frozen=True)
class PowerFit:
    """Run-power profile, temperature slope and measured low-power states."""

    profile_w: tuple[float, ...]
    per_c_w: float
    residual_sd_w: float
    idle_w: float
    off_w: float
    metered_scope: MeteredScope

    def __post_init__(self) -> None:
        if (
            not self.profile_w
            or not np.isfinite([*self.profile_w, self.per_c_w]).all()
            or min(self.profile_w) < 0
            or min(self.idle_w, self.off_w, self.residual_sd_w) < 0
        ):
            raise HvacError("invalid electrical power coefficients")

    def run_power(self, minute: ArrayLike, outdoor_c: ArrayLike) -> NDArray[np.float64]:
        """Mean electrical power at a minute of a run.

        Args:
            minute: Minutes since the run started (0 is the first minute).
            outdoor_c: Outdoor temperature.

        Returns:
            Nonnegative watts in the declared meter scope.
        """
        profile = np.asarray(self.profile_w)
        k = np.clip(np.asarray(minute), 0, len(profile) - 1).astype(np.int64)
        value = profile[k] + self.per_c_w * (np.asarray(outdoor_c, dtype=float) - 7)
        return np.asarray(np.maximum(value, 0.0))


def fit_power(
    frame: pd.DataFrame,
    mode_events: pd.DataFrame,
    cfg: HvacModelConfig,
    scope: MeteredScope,
) -> PowerFit:
    """Fit the run-power profile and temperature slope, plus idle/off power.

    Args:
        frame: Regular minute frame with ``power_w``, cleaned ``running``
            and ``outdoor_c``.
        mode_events: Causal command history (for off-mode standby).
        cfg: Mode configuration.
        scope: Components measured by the meter.

    Returns:
        The fitted power model.

    Raises:
        ValueError: If running or idle power cannot be estimated.
    """
    running = frame.running
    segment = running.ne(running.shift()).cumsum()
    minute = running.groupby(segment).cumcount().to_numpy()
    # A run whose start is a gap has an unknown minute index; exclude it.
    known_start = running.shift().notna().groupby(segment).transform("first").to_numpy()
    use = (
        running.eq(1).to_numpy()
        & known_start
        & frame.power_w.notna().to_numpy()
        & np.isfinite(frame.outdoor_c.to_numpy(dtype=float))
    )
    if use.sum() < MIN_RUN_MINUTES_FOR_FIT:
        raise HvacError("insufficient_run_power")
    k = minute[use]
    p = frame.power_w.to_numpy(dtype=float)[use]
    # Winsorize metering spikes so a single bad sample cannot move a mean.
    p = np.minimum(p, 1.5 * float(np.median(p)))
    t = frame.outdoor_c.to_numpy(dtype=float)[use] - 7
    counts = np.bincount(np.minimum(k, MAX_PROFILE_MINUTES))
    supported = np.flatnonzero(counts >= MIN_RUNS_PER_PROFILE_MINUTE)
    last = int(supported.max()) if len(supported) else 0
    k = np.minimum(k, last)
    # Temperature slope within minute-in-run cells (demeaned by minute).
    centred_p = p - pd.Series(p).groupby(k).transform("median").to_numpy()
    centred_t = t - pd.Series(t).groupby(k).transform("mean").to_numpy()
    slope = float(huber_regression(centred_t[:, None], centred_p)[0])
    adjusted = p - slope * t
    # Means, not medians: every run ends in a partial minute, and only the mean
    # profile reproduces measured run energy (medians overstate it by ~5 %).
    profile = pd.Series(adjusted).groupby(k).mean().reindex(range(last + 1))
    profile = profile.interpolate(limit_direction="both")
    residual = adjusted - profile.to_numpy()[k]
    modes = modes_at(pd.DatetimeIndex(frame.index), mode_events, cfg.reference_mode)
    off = modes.isin(cfg.off_modes).to_numpy()
    standby = frame.power_w.to_numpy(dtype=float)
    idle = standby[running.eq(0).to_numpy() & ~off]
    idle = idle[np.isfinite(idle)]
    if not len(idle):
        raise HvacError("no_measured_idle_power")
    off_power = standby[running.eq(0).to_numpy() & off]
    off_power = off_power[np.isfinite(off_power)]
    idle_w = float(np.median(idle))
    return PowerFit(
        tuple(float(v) for v in profile),
        slope,
        float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        idle_w,
        float(np.median(off_power)) if len(off_power) else idle_w,
        scope,
    )
