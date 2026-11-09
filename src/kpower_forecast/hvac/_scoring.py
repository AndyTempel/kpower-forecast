"""Full-ensemble per-origin scores; persisted rows contain no simulated paths."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._baselines import BaselineForecast
from ._stats import sample_crps
from .aggregate import _integral
from .evaluation import (
    actual_intervals,
    ks_distance,
    lead_bucket,
    pinball,
    regular_grid,
)
from .types import GridInterval, HvacError, HvacForecast


def _samples(
    forecast: HvacForecast, grid: tuple[GridInterval, ...]
) -> NDArray[np.float64]:
    left = np.array(
        [
            (step.start - forecast.minute_paths.start).total_seconds() / 60
            for step in grid
        ]
    )
    right = left + np.array([step.minutes for step in grid])
    if left[0] < 0 or right[-1] > forecast.minute_paths.power_w.shape[1]:
        raise HvacError("score grid outside the retained paths")
    return _integral(forecast.minute_paths.power_w, left, right) / 60000


def _identity(forecast: HvacForecast, step: GridInterval) -> dict[str, object]:
    lead = (step.start - forecast.origin).total_seconds() / 60
    return {
        "origin": forecast.origin,
        "interval_start": step.start,
        "minutes": step.minutes,
        "lead_min": lead,
        "bucket": lead_bucket(lead),
        "tier": forecast.tier.value,
    }


def quarter_scores(
    forecast: HvacForecast,
    actual: pd.DataFrame,
    baselines: Mapping[str, BaselineForecast],
    *,
    threshold_w: float | None,
    hours: int = 72,
) -> pd.DataFrame:
    """Compute M1/M1q on quarters from all paths and matched qualified actuals.

    Args:
        forecast: Complete in-process ensemble, never representative paths.
        actual: Qualified minute outcomes, separate from model features.
        baselines: Registered quarter distributions on this exact grid.
        threshold_w: Causal fitted running guard for actual state.
        hours: Whole-hour scoring horizon within the retained paths.

    Returns:
        One row per model and quarter, including explicit eligibility. Unknown
        scores remain NaN; baseline comparisons must join identical identities.

    Raises:
        ValueError: For incompatible grids or malformed evidence.
    """
    grid = regular_grid(forecast.origin, minutes=15, hours=hours)
    observed = actual_intervals(actual, grid, threshold_w=threshold_w)
    stats = forecast.stats(grid)
    quantiles = np.stack([stats.q10_kwh, stats.q50_kwh, stats.q90_kwh])
    distributions = {"cycle": (stats.expected_kwh, quantiles)}
    distributions.update(
        {
            key: (value.expected_kwh, value.quantiles_kwh)
            for key, value in baselines.items()
        }
    )
    samples = _samples(forecast, grid)
    records: list[dict[str, object]] = []
    for name, (expected, bounds) in distributions.items():
        if expected.shape != (len(grid),) or bounds.shape != (3, len(grid)):
            raise HvacError("baseline does not match the quarter grid")
        use = observed.eligible_energy & np.isfinite(bounds).all(axis=0)
        use &= np.isfinite(expected)
        score = np.full(len(grid), np.nan)
        score[use] = pinball(observed.energy_kwh[use], bounds[:, use])
        crps = np.full(len(grid), np.nan)
        if name == "cycle":
            crps[use] = sample_crps(samples[:, use], observed.energy_kwh[use])
        for i, step in enumerate(grid):
            records.append(
                {
                    **_identity(forecast, step),
                    "model": name,
                    "eligible": bool(use[i]),
                    "actual_kwh": observed.energy_kwh[i],
                    "expected_kwh": expected[i],
                    "q10_kwh": bounds[0, i],
                    "q50_kwh": bounds[1, i],
                    "q90_kwh": bounds[2, i],
                    "pinball": score[i],
                    "crps": crps[i],
                }
            )
    return pd.DataFrame.from_records(records)


def run_scores(
    forecast: HvacForecast,
    actual: pd.DataFrame,
    *,
    threshold_w: float | None,
    hours: int = 72,
    r1c: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Compute M2 on separate five- and fifteen-minute grids, without summing odds.

    Args:
        forecast: Full ensemble whose statistics are recomputed for each grid.
        actual: Qualified outcomes.
        threshold_w: Causal actual-state guard; missing state stays unknown.
        hours: Quarter-grid horizon; five-minute scores cover at most two hours.
        r1c: Optional B5 first-hour five-minute probabilities on identical pairs.

    Returns:
        Probability, actual outcome and raw Brier per model/grid interval.

    Raises:
        ValueError: If B5 does not cover exactly twelve first-hour intervals.
    """
    if r1c is not None and r1c.shape != (12,):
        raise HvacError("B5 must have twelve first-hour probabilities")
    records: list[dict[str, object]] = []
    for minutes, span in ((5, min(hours, 2)), (15, hours)):
        grid = regular_grid(forecast.origin, minutes=minutes, hours=span)
        outcomes = actual_intervals(actual, grid, threshold_w=threshold_w)
        probability = forecast.stats(grid).run_probability
        predictions = {
            "cycle": (
                probability if probability is not None else np.full(len(grid), np.nan)
            )
        }
        if minutes == 5 and r1c is not None:
            predictions["B5"] = np.concatenate([r1c, np.full(len(grid) - 12, np.nan)])
        for name, p in predictions.items():
            if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
                raise HvacError("invalid comparison probability")
            for i, step in enumerate(grid):
                eligible = bool(outcomes.eligible_run[i] and np.isfinite(p[i]))
                records.append(
                    {
                        **_identity(forecast, step),
                        "model": name,
                        "eligible": eligible,
                        "probability": p[i],
                        "n_paths": len(forecast.minute_paths.power_w),
                        "actual_run": outcomes.ran[i],
                        "brier": (p[i] - outcomes.ran[i]) ** 2 if eligible else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def energy_scores(
    forecast: HvacForecast,
    actual: pd.DataFrame,
    baselines: Mapping[str, BaselineForecast],
    *,
    hours: int = 72,
) -> pd.DataFrame:
    """Score M6 complete windows and distinct observed-footprint diagnostics.

    Args:
        forecast: All minute paths; only their mean is used for energy bias/MAE.
        actual: Qualified minute power, with missing outcomes retained.
        baselines: Quarter predictions held constant inside each quarter only
            for the explicitly labelled partial-data diagnostic.
        hours: Whole-hour horizon, including the separate 120-hour M6 pass.

    Returns:
        Non-overlapping 1/6/24-hour windows. Official error is NaN unless complete;
        partial errors compare exactly the observed minutes without rescaling.

    Raises:
        ValueError: For incompatible baseline lengths or a non-minute origin.
    """
    if forecast.origin.second or forecast.origin.microsecond:
        raise HvacError("offline energy scores require a minute-aligned origin")
    index = pd.date_range(forecast.origin, periods=hours * 60, freq="min")
    observed = actual.reindex(index).power_w.to_numpy(dtype=float)
    offset = int((forecast.origin - forecast.minute_paths.start).total_seconds() / 60)
    mean = np.mean(
        forecast.minute_paths.power_w[:, offset : offset + len(index)], axis=0
    )
    predictions = {"cycle": np.asarray(mean, dtype=float)}
    for name, baseline in baselines.items():
        if baseline.expected_kwh.shape != (hours * 4,):
            raise HvacError("energy baseline does not match the horizon")
        predictions[name] = np.repeat(baseline.expected_kwh * 4000, 15)
    return _energy_windows(
        forecast.origin, index, observed, predictions, cycle_method="minute_path_mean"
    )


def _energy_windows(
    origin: datetime,
    index: pd.DatetimeIndex,
    observed: NDArray[np.float64],
    predictions: Mapping[str, NDArray[np.float64]],
    *,
    cycle_method: str,
) -> pd.DataFrame:
    known = np.isfinite(observed)
    if any(value.shape != observed.shape for value in predictions.values()):
        raise HvacError("energy predictions and actual footprint differ")
    records: list[dict[str, object]] = []
    for duration in (1, 6, 24):
        length = duration * 60
        for left in range(0, len(index) - length + 1, length):
            right = left + length
            mask = known[left:right]
            coverage = float(np.mean(mask))
            for name, predicted in predictions.items():
                values = predicted[left:right]
                complete = coverage == 1 and bool(np.isfinite(values).all())
                partial = bool(mask.any() and np.isfinite(values[mask]).all())
                actual_partial = float(np.sum(observed[left:right][mask]) / 60000)
                predicted_partial = (
                    float(np.sum(values[mask]) / 60000) if partial else np.nan
                )
                expected = float(np.sum(values) / 60000)
                records.append(
                    {
                        "origin": origin,
                        "interval_start": index[left].to_pydatetime(),
                        "duration_h": duration,
                        "model": name,
                        "eligible": complete,
                        "coverage": coverage,
                        "observed_minutes": int(mask.sum()),
                        "actual_kwh": actual_partial if complete else np.nan,
                        "expected_kwh": expected,
                        "error_kwh": expected - actual_partial if complete else np.nan,
                        "partial_actual_kwh": actual_partial if mask.any() else np.nan,
                        "partial_expected_kwh": predicted_partial,
                        "partial_error_kwh": predicted_partial - actual_partial,
                        "partial_method": (
                            cycle_method
                            if name == "cycle"
                            else "quarter_mean_piecewise_constant"
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _spell_lengths(states: NDArray[np.bool_], value: bool) -> NDArray[np.float64]:
    # Edge spells are censored; comparing their truncated lengths would bias KS.
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    if len(changes) < 2:
        return np.empty(0)
    keep = states[changes[:-1]] == value
    return np.asarray(np.diff(changes)[keep], dtype=float)


def cadence_scores(
    forecast: HvacForecast,
    actual: pd.DataFrame,
    *,
    threshold_w: float | None,
    hours: int = 72,
) -> pd.DataFrame:
    """Compute M5 count CRPS and complete-day uncensored run/idle length KS.

    Args:
        forecast: All paths with optional compressor states and start indicators.
        actual: Qualified minute outcomes including the minute before origin.
        threshold_w: Causal state guard; explicit qualified flags take precedence.
        hours: Scoring horizon, in whole hours.

    Returns:
        Six-hour count and daily KS rows, explicitly unscorable on gaps or T0.

    Raises:
        ValueError: For invalid binary state evidence.
    """
    index = pd.date_range(
        forecast.origin - timedelta(minutes=1), periods=hours * 60 + 1, freq="min"
    )
    rows = actual.reindex(index)
    power = rows.power_w.to_numpy(dtype=float)
    state = np.full(len(rows), np.nan)
    if threshold_w is not None:
        state[np.isfinite(power)] = (power[np.isfinite(power)] > threshold_w).astype(
            float
        )
    if "running" in rows:
        flag = rows.running.to_numpy(dtype=float)
        if not np.isin(flag[np.isfinite(flag)], [0, 1]).all():
            raise HvacError("invalid cadence state flag")
        use = np.isfinite(flag) & np.isfinite(power)
        state[use] = flag[use]
    offset = int((forecast.origin - forecast.minute_paths.start).total_seconds() / 60)
    records: list[dict[str, object]] = []
    for span in (6, 24):
        length = span * 60
        for left in range(0, hours * 60 - length + 1, length):
            right = left + length
            segment = state[left : right + 1]
            paths = forecast.minute_paths
            eligible = bool(
                np.isfinite(segment).all()
                and paths.running is not None
                and paths.starts is not None
            )
            count_crps = run_ks = idle_ks = np.nan
            actual_count = np.nan
            if eligible:
                assert paths.starts is not None and paths.running is not None
                actual_count = float(np.sum((segment[1:] == 1) & (segment[:-1] == 0)))
                if span == 6:
                    counts = np.sum(
                        paths.starts[:, offset + left : offset + right], axis=1
                    )
                    count_crps = float(sample_crps(counts[:, None], [actual_count])[0])
                else:
                    simulated = paths.running[:, offset + left : offset + right]
                    for value in (True, False):
                        lengths = np.concatenate(
                            [_spell_lengths(path, value) for path in simulated]
                        )
                        observed_lengths = _spell_lengths(segment[1:] > 0, value)
                        distance = ks_distance(lengths, observed_lengths)
                        if value:
                            run_ks = distance if distance is not None else np.nan
                        else:
                            idle_ks = distance if distance is not None else np.nan
            records.append(
                {
                    "origin": forecast.origin,
                    "interval_start": index[left + 1].to_pydatetime(),
                    "duration_h": span,
                    "eligible": eligible,
                    "actual_starts": actual_count,
                    "start_count_crps": count_crps,
                    "run_length_ks": run_ks,
                    "idle_length_ks": idle_ks,
                }
            )
    return pd.DataFrame.from_records(records)
