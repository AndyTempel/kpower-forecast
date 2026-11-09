"""Causal development/live scoring primitives, with unknown outcomes excluded.

Development scores are bug/regression evidence, never proof of unseen skill.
Full ensemble statistics stay in process; display representatives are not scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .aggregate import _integral, _touch
from .types import GridInterval, HvacError, utc, validate_grid, validate_minute_index

LEAD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-15m", 0, 15),
    ("15-30m", 15, 30),
    ("30-60m", 30, 60),
    ("1-3h", 60, 180),
    ("3-6h", 180, 360),
    ("6-24h", 360, 1440),
    ("24-48h", 1440, 2880),
    ("48-72h", 2880, 4320),
)


def lead_bucket(minutes: float) -> str | None:
    """Return the pre-registered bucket for an interval's start lead.

    Args:
        minutes: Lead from the actual forecast origin to interval start.

    Returns:
        Registered bucket or None outside the 72-hour metric horizon.
    """
    return next(
        (name for name, left, right in LEAD_BUCKETS if left <= minutes < right), None
    )


def mixed_grid(origin: datetime, *, hours: int = 72) -> tuple[GridInterval, ...]:
    """Construct five-minute first-two-hour output, then quarter-hour intervals.

    Args:
        origin: Actual UTC origin, also the offline output-grid start.
        hours: Positive whole-hour horizon, at most five days.

    Returns:
        A contiguous mixed grid. EMS instead supplies its upcoming-quarter start
        while retaining its earlier actual prediction origin.

    Raises:
        ValueError: For naive time or an invalid horizon.
    """
    origin = utc(origin)
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 120:
        raise HvacError("evaluation horizon must be 1..120 whole hours")
    near = min(hours * 60, 120)
    return tuple(
        [
            GridInterval(start=origin + timedelta(minutes=i), minutes=5)
            for i in range(0, near, 5)
        ]
        + [
            GridInterval(start=origin + timedelta(minutes=i), minutes=15)
            for i in range(near, hours * 60, 15)
        ]
    )


def regular_grid(
    origin: datetime, *, minutes: int, hours: int
) -> tuple[GridInterval, ...]:
    """Construct a supported regular scoring grid on the same retained paths.

    Args:
        origin: UTC grid start.
        minutes: Supported interval length, dividing the whole horizon.
        hours: Positive whole-hour horizon.

    Returns:
        Contiguous scoring intervals.

    Raises:
        ValueError: If the grid duration is unsupported or does not divide the span.
    """
    origin = utc(origin)
    if minutes not in (1, 5, 15, 30, 60) or hours <= 0 or (hours * 60) % minutes:
        raise HvacError("invalid scoring grid")
    return tuple(
        GridInterval.model_validate(
            {"start": origin + timedelta(minutes=i), "minutes": minutes}
        )
        for i in range(0, hours * 60, minutes)
    )


def pinball(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> NDArray[np.float64]:
    """Compute mean pinball over registered quantiles for each matched interval.

    Args:
        actual: Finite actual energy per interval.
        predicted: Quantiles along axis zero, intervals along axis one.
        quantiles: Probabilities corresponding to predicted rows.

    Returns:
        One mean quantile score per interval, in the actuals' units.

    Raises:
        ValueError: For malformed or nonfinite evidence; callers exclude unknowns.
    """
    y, p, q = (
        np.asarray(actual, dtype=float),
        np.asarray(predicted, dtype=float),
        np.asarray(quantiles, dtype=float),
    )
    if (
        y.ndim != 1
        or p.shape != (len(q), len(y))
        or q.ndim != 1
        or not len(q)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or (q < 0).any()
        or (q > 1).any()
        or (np.diff(q) <= 0).any()
    ):
        raise HvacError("invalid pinball inputs")
    error = y[None, :] - p
    return np.asarray(
        np.mean(np.maximum(q[:, None] * error, (q[:, None] - 1) * error), axis=0)
    )


@dataclass(frozen=True)
class ActualIntervals:
    """Qualified interval actuals; missing/invalid outcomes remain NaN, never zero."""

    energy_kwh: NDArray[np.float64]
    ran: NDArray[np.float64]
    eligible_energy: NDArray[np.bool_]
    eligible_run: NDArray[np.bool_]


def actual_intervals(
    frame: pd.DataFrame, grid: tuple[GridInterval, ...], *, threshold_w: float | None
) -> ActualIntervals:
    """Score only intervals wholly covered by qualified minute power/state.

    Args:
        frame: Availability-qualified actual target, with optional binary flag.
        grid: Requested scoring intervals, all using their own boundaries.
        threshold_w: Causal fit's running guard; None leaves unknown state unknown.

    Returns:
        Actual energy/run outcomes and explicit eligibility masks. Scratch prefix
        sums substitute zero only under masks; every affected interval is excluded.
        This never fills the input frame or turns a missing minute into an idle.

    Raises:
        ValueError: For invalid minute identity, grid or binary evidence.
    """
    validate_minute_index(frame)
    validate_grid(grid)
    start = pd.Timestamp(grid[0].start).floor("min")
    end = pd.Timestamp(grid[-1].end).ceil("min")
    index = pd.date_range(start, end, freq="min", inclusive="left")
    rows = frame.reindex(index)
    power = rows.power_w.to_numpy(dtype=float)
    known = np.isfinite(power)
    if threshold_w is not None and (not np.isfinite(threshold_w) or threshold_w < 0):
        raise HvacError("invalid actual running guard")
    running = (
        np.where(known, (power > threshold_w).astype(float), np.nan)
        if threshold_w is not None
        else np.full(len(power), np.nan)
    )
    if "running" in rows:
        flag = rows.running.to_numpy(dtype=float)
        if not np.isin(flag[np.isfinite(flag)], [0.0, 1.0]).all():
            raise HvacError("actual running flag must be binary or unknown")
        valid_flag = np.isfinite(flag) & known
        running[valid_flag] = flag[valid_flag]
    left = np.array(
        [(item.start - start.to_pydatetime()).total_seconds() / 60 for item in grid]
    )
    right = left + np.array([item.minutes for item in grid])
    observed = _integral(known[None, :], left, right)[0]
    eligible = np.isclose(observed, right - left, rtol=0, atol=1e-9)
    energy = _integral(np.where(known, power, 0.0)[None, :], left, right)[0] / 60000
    energy[~eligible] = np.nan
    run_known = np.isfinite(running)
    eligible_run = np.isclose(
        _integral(run_known[None, :], left, right)[0], right - left, rtol=0, atol=1e-9
    )
    ran = _touch(
        np.where(run_known, running > 0, False)[None, :],
        np.floor(left).astype(np.int64),
        np.ceil(right).astype(np.int64),
    )
    ran[~eligible_run] = np.nan
    for value in (energy, ran, eligible, eligible_run):
        value.setflags(write=False)
    return ActualIntervals(energy, ran, eligible, eligible_run)


def reliability_bins(
    probability: NDArray[np.float64], actual: NDArray[np.float64]
) -> list[dict[str, float | int | None]]:
    """Return ten reliability bins using the original, unclipped probabilities.

    Args:
        probability: Finite probabilities for eligible matched outcomes.
        actual: Binary actual run events on those same pairs.

    Returns:
        Bin counts, mean probability and observed rate; empty bins retain None.

    Raises:
        ValueError: For invalid probabilities or outcomes.
    """
    p, y = _binary_inputs(probability, actual)
    which = np.minimum((p * 10).astype(int), 9)
    result = []
    for i in range(10):
        use = which == i
        result.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "n": int(use.sum()),
                "predicted": float(p[use].mean()) if use.any() else None,
                "observed": float(y[use].mean()) if use.any() else None,
            }
        )
    return result


def _binary_inputs(
    probability: NDArray[np.float64], actual: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p, y = np.asarray(probability, dtype=float), np.asarray(actual, dtype=float)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.isin(y, [0.0, 1.0]).all()
    ):
        raise HvacError("invalid probability/outcome pairs")
    return p, y


@dataclass(frozen=True)
class CalibrationFit:
    """Unpenalized logistic calibration; an undefined fit is never slope one."""

    intercept: float | None
    slope: float | None
    observations: int
    reason: str | None


def calibration_slope(
    probability: NDArray[np.float64],
    actual: NDArray[np.float64],
    *,
    n_paths: int | NDArray[np.int64],
) -> CalibrationFit:
    """Fit the maintainer-approved logistic calibration definition for A5.

    Args:
        probability: Original eligible run probabilities (not display paths).
        actual: Binary outcomes on identical pairs.
        n_paths: Ensemble size, defining boundary clipping at 0.5/n_paths.

    Returns:
        Free intercept and slope, or an explicit unidentifiable/separated reason.
        Clipping is only for this regression; Brier/reliability use raw inputs.

    Raises:
        ValueError: For malformed probabilities, actuals or path count.
    """
    p, y = _binary_inputs(probability, actual)
    sizes = np.asarray(n_paths)
    if (
        sizes.dtype.kind not in "iu"
        or sizes.dtype.kind == "b"
        or sizes.shape not in ((), p.shape)
        or (sizes < 2).any()
    ):
        raise HvacError("calibration requires at least two paths")
    if len(p) < 3 or np.unique(y).size < 2:
        return CalibrationFit(None, None, len(p), "insufficient_outcome_variation")
    clipped = np.clip(p, 0.5 / sizes, 1 - 0.5 / sizes)
    logit = np.log(clipped) - np.log1p(-clipped)
    x = np.column_stack([np.ones(len(p)), logit])
    if np.linalg.matrix_rank(x) < 2:
        return CalibrationFit(None, None, len(p), "constant_probability")
    if (
        logit[y == 0].max() <= logit[y == 1].min()
        or logit[y == 1].max() <= logit[y == 0].min()
    ):
        return CalibrationFit(None, None, len(p), "separated_or_singular")
    beta = np.zeros(2)
    for _ in range(100):
        score = x @ beta
        expected = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
        hessian = x.T @ ((expected * (1 - expected))[:, None] * x)
        gradient = x.T @ (y - expected)
        if not np.isfinite(hessian).all() or np.linalg.cond(hessian) > 1e12:
            return CalibrationFit(None, None, len(p), "separated_or_singular")
        update = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(update)) < 1e-8:
            return CalibrationFit(float(beta[0]), float(beta[1]), len(p), None)
        # Halving maximizes the binomial likelihood without a penalty that could
        # manufacture a slope inside the acceptance range under separation.
        old = float(np.sum(y * score - np.logaddexp(0, score)))
        factor = 1.0
        for _ in range(30):
            trial = beta + factor * update
            trial_score = x @ trial
            value = float(np.sum(y * trial_score - np.logaddexp(0, trial_score)))
            if np.isfinite(value) and value >= old - 1e-10:
                beta = trial
                break
            factor *= 0.5
        else:
            return CalibrationFit(None, None, len(p), "logistic_nonconvergence")
        if np.max(np.abs(beta)) > 1000:
            return CalibrationFit(None, None, len(p), "separated_or_singular")
    return CalibrationFit(None, None, len(p), "logistic_nonconvergence")


def ks_distance(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float | None:
    """Compute empirical two-sample KS distance, excluding no observations silently.

    Args:
        first: Finite sampled spell lengths.
        second: Finite observed spell lengths.

    Returns:
        Maximum empirical CDF difference, or None when either sample is empty.

    Raises:
        ValueError: For nonfinite input.
    """
    a, b = (
        np.sort(np.asarray(first, dtype=float)),
        np.sort(np.asarray(second, dtype=float)),
    )
    if (
        a.ndim != 1
        or b.ndim != 1
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise HvacError("invalid KS samples")
    if not len(a) or not len(b):
        return None
    at = np.union1d(a, b)
    return float(
        np.max(
            np.abs(
                np.searchsorted(a, at, side="right") / len(a)
                - np.searchsorted(b, at, side="right") / len(b)
            )
        )
    )
