"""Buffer-temperature nowcast of the current idle.

A fixed-speed unit restarts when its control (buffer) temperature falls to
``setpoint - hysteresis``. During an idle the buffer cools almost linearly, so
the remaining idle is about ``gap / cooling rate``. The fit only calibrates that
physical estimate (median log ratio and its spread) on complete idles; without
enough evidence the nowcast is absent and the duration model is used alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SLOPE_WINDOW_MIN = 8
MIN_SLOPE_POINTS = 6
MIN_COOLING_C_PER_MIN = 0.02
MAX_ESTIMATE_MIN = 300.0
MIN_SAMPLES = 20
MIN_SPELLS = 6
SAMPLE_EVERY_MIN = 5
MIN_SIGMA = 0.08


@dataclass(frozen=True)
class WaterFit:
    """Calibration of the physical remaining-idle estimate (log scale)."""

    bias: float
    sigma: float
    samples: int


def water_estimates(frame: pd.DataFrame) -> pd.Series:
    """Remaining idle minutes after each minute from the buffer gap and cooling rate.

    Args:
        frame: Regular minute frame with ``control_c``, ``control_setpoint_c`` and
            ``control_hysteresis_c``; missing values stay missing.

    Returns:
        Minutes until the restart threshold, NaN where the buffer is not cooling
        or the evidence is incomplete.
    """
    columns = ("control_c", "control_setpoint_c", "control_hysteresis_c")
    if any(c not in frame for c in columns):
        return pd.Series(np.nan, index=frame.index)
    y = pd.to_numeric(frame.control_c, errors="coerce").astype(float)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index).where(y.notna())
    w = SLOPE_WINDOW_MIN
    count = y.rolling(w, min_periods=MIN_SLOPE_POINTS).count()
    mean_x = x.rolling(w, min_periods=MIN_SLOPE_POINTS).mean()
    mean_y = y.rolling(w, min_periods=MIN_SLOPE_POINTS).mean()
    cov = (x * y).rolling(w, min_periods=MIN_SLOPE_POINTS).mean() - mean_x * mean_y
    var = (x * x).rolling(w, min_periods=MIN_SLOPE_POINTS).mean() - mean_x**2
    slope = (cov / var.where(var > 0)).where(count >= MIN_SLOPE_POINTS)
    setpoint = pd.to_numeric(frame.control_setpoint_c, errors="coerce")
    hysteresis = pd.to_numeric(frame.control_hysteresis_c, errors="coerce")
    threshold = setpoint - hysteresis
    gap = (y - threshold).clip(lower=0.0)
    estimate = gap / (-slope)
    usable = slope.lt(-MIN_COOLING_C_PER_MIN) & y.notna() & threshold.notna()
    return estimate.where(usable).clip(lower=1.0, upper=MAX_ESTIMATE_MIN)


def fit_water(frame: pd.DataFrame, spells: pd.DataFrame) -> WaterFit | None:
    """Calibrate the estimate on complete idles, sampled every few minutes.

    Args:
        frame: Qualified minute frame (see :func:`water_estimates`).
        spells: Extracted spells with ``kind``, ``start``, ``minutes``, ``censored``.

    Returns:
        Median log ratio actual/estimate and its robust spread, or None when
        fewer than the required samples or spells have buffer evidence.
    """
    estimates = water_estimates(frame)
    if estimates.notna().sum() == 0:
        return None
    idles = spells.loc[spells.kind.eq("idle") & ~spells.censored]
    ratios: list[float] = []
    used = 0
    for start, minutes in zip(idles.start, idles.minutes, strict=True):
        length = int(minutes)
        found = False
        for k in range(SAMPLE_EVERY_MIN, length - 1, SAMPLE_EVERY_MIN):
            at = pd.Timestamp(start) + pd.Timedelta(minutes=k - 1)
            value = estimates.get(at, np.nan)
            if np.isfinite(value):
                ratios.append(float(np.log((length - k) / value)))
                found = True
        used += found
    if len(ratios) < MIN_SAMPLES or used < MIN_SPELLS:
        return None
    values = np.asarray(ratios)
    bias = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values - bias)))
    return WaterFit(bias, max(sigma, MIN_SIGMA), len(values))


def current_water(recent: pd.DataFrame, origin: pd.Timestamp) -> float | None:
    """Physical remaining idle at the origin from the newest buffer evidence.

    Args:
        recent: Availability-sliced recent minutes.
        origin: Forecast origin.

    Returns:
        Minutes from the origin, or None without a fresh cooling estimate.
    """
    estimates = water_estimates(recent).dropna()
    if estimates.empty:
        return None
    last = estimates.index[-1]
    if last != recent.index[-1]:
        return None
    age = (origin - (last + pd.Timedelta(minutes=1))).total_seconds() / 60
    return float(max(float(estimates.iloc[-1]) - max(age, 0.0), 0.0))
