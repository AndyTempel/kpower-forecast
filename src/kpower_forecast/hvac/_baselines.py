"""Causal registered forecast baselines, retaining missing legacy evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bundle import BundleSnapshot
from .evaluation import actual_intervals, regular_grid
from .frames import policy_number, prepare_minutes
from .types import GridInterval, HvacError, HvacModelConfig


@dataclass(frozen=True)
class BaselineForecast:
    """Quarter energy statistics and the plan-defined M3 deterministic power path."""

    expected_kwh: NDArray[np.float64]
    quantiles_kwh: NDArray[np.float64]
    planner_w: NDArray[np.float64]
    diagnostics: dict[str, int]


def _point(
    values_w: NDArray[np.float64], grid: tuple[GridInterval, ...], reason: str
) -> BaselineForecast:
    energy = values_w * np.array([step.minutes for step in grid]) / 60000
    return BaselineForecast(
        energy,
        np.repeat(energy[None, :], 3, axis=0),
        values_w,
        {reason: int(np.isfinite(energy).sum())},
    )


def simple_baselines(
    view: BundleSnapshot, cfg: HvacModelConfig, grid: tuple[GridInterval, ...]
) -> dict[str, BaselineForecast]:
    """Build B1 trailing mean and B2 repeated yesterday pattern without future data.

    Args:
        view: One origin-availability snapshot.
        cfg: Same resolved quality policy and ingestion lag as the evaluated model.
        grid: Quarter-hour grid starting at origin or its upcoming quarter.

    Returns:
        Point-distribution naive baselines. B2 repeats the latest 24-hour pattern
        for later days; it never reads the actual day just before a future target.
        Missing yesterday quarters remain unknown, including the ingestion tail.

    Raises:
        ValueError: For a non-quarter grid or a start outside the next quarter.
    """
    if (
        not grid
        or not timedelta(0) <= grid[0].start - view.origin < timedelta(minutes=15)
        or any(step.minutes != 15 for step in grid)
    ):
        raise HvacError("naive baselines require an origin-aligned quarter grid")
    frame = prepare_minutes(
        view.training.minutes,
        cfg,
        as_of=view.origin,
        training=False,
        policy=view.training.history_policy,
        epoch_start=view.training.epoch_start,
    )
    past = frame.loc[frame.index >= pd.Timestamp(view.origin) - pd.Timedelta(days=1)]
    density = past.power_w.notna().sum() / 1440
    enough = density >= policy_number(view.training.history_policy, "_MIN_DENSITY")
    mean = float(past.power_w.mean()) if enough else np.nan
    b1 = _point(np.full(len(grid), mean), grid, "eligible_trailing_mean_intervals")
    yesterday = regular_grid(grid[0].start - timedelta(days=1), minutes=15, hours=24)
    actual = actual_intervals(frame, yesterday, threshold_w=None)
    phase = np.array(
        [int((step.start - grid[0].start).total_seconds() // 900) % 96 for step in grid]
    )
    previous_power = actual.energy_kwh[phase] * 4000
    b2 = _point(previous_power, grid, "eligible_yesterday_intervals")
    return {"B1": b1, "B2": b2}


def _legacy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "lower_w" not in result and "q10_w" in result:
        result["lower_w"] = result.q10_w
    if "upper_w" not in result and "q90_w" in result:
        result["upper_w"] = result.q90_w
    return result


def _raised_flags(
    rows: pd.DataFrame, metadata: pd.Series | None
) -> NDArray[np.float64]:
    """Count-check a labelled development proxy; never use it for live scoring."""
    unknown = np.full(len(rows), np.nan)
    if metadata is None:
        return unknown
    applied = metadata.get("applied")
    raised = metadata.get("raised_upper_intervals")
    if pd.isna(applied) or pd.isna(raised) or isinstance(raised, bool):
        return unknown
    count = int(raised)
    if count < 0 or float(raised) != count:
        return unknown
    if not bool(applied) or count == 0:
        return np.zeros(len(rows))
    horizon = metadata.get("horizon")
    complete_size = (
        len(rows) == 5 * 24 * 4 if horizon == "daily" else len(rows) in (4, 5)
    )
    times = pd.DatetimeIndex(rows.interval_start).sort_values()
    contiguous = bool((times[1:] - times[:-1] == pd.Timedelta(minutes=15)).all())
    if not complete_size or not contiguous:
        return unknown
    level = metadata.get("run_power_w")
    if level is None or pd.isna(level) or not np.isfinite(float(level)):
        return unknown
    # Publisher rounds run metadata to0.1W, energy to1e-6kWh (0.004W at15min).
    # Count agreement is essential: a truncated/colliding old version is not
    # silently completed from later publications or guessed flag positions.
    match = np.isclose(
        rows.upper_w.to_numpy(dtype=float),
        float(level),
        rtol=0,
        atol=0.05 + 0.002 + 1e-9,
    )
    if int(match.sum()) != count:
        return unknown
    return np.asarray(match, dtype=float)


def legacy_baseline(
    view: BundleSnapshot, grid: tuple[GridInterval, ...], *, live: bool = False
) -> BaselineForecast:
    """Reconstruct available B3 intervals with the owning hourly-then-daily priority.

    Args:
        view: Causal old publications and band metadata, separate from actuals.
        grid: Quarter scoring grid.
        live: Read the archived legacy shadow instead of development served runs.

    Returns:
        Quantile mapping q10=lower,q50=expected,q90=upper. M3 uses upper only
        where run-awareness raised it, as pre-registered. This is a forecast
        proxy, not a full legacy MPC replay (which also had other control rules).
        Missing/ambiguous metadata or lost intervals remain unscorable.
    """
    source = view.baselines.get(
        "legacy_shadow.parquet" if live else "served_forecast.parquet", pd.DataFrame()
    )
    if source.empty:
        empty = np.full(len(grid), np.nan)
        return BaselineForecast(
            empty,
            np.full((3, len(grid)), np.nan),
            empty.copy(),
            {"missing_legacy": len(grid)},
        )
    source = _legacy_columns(source)
    metadata = view.baselines.get("legacy_awareness.parquet", pd.DataFrame())
    expected = np.full(len(grid), np.nan)
    quantiles = np.full((3, len(grid)), np.nan)
    planner = np.full(len(grid), np.nan)
    selected = np.zeros(len(grid), bool)
    blocked = np.zeros(len(grid), bool)
    diagnostics = {
        "legacy_quantile_intervals": 0,
        "legacy_m3_intervals": 0,
        "legacy_missing_or_ambiguous_band": 0,
    }
    for horizon in ("hourly", "daily"):
        rows = source.loc[source.horizon.eq(horizon)]
        meta = metadata.loc[metadata.horizon.eq(horizon)] if len(metadata) else metadata
        origins = []
        if len(rows):
            origins.append(pd.Timestamp(rows.origin.max()))
        if len(meta):
            origins.append(pd.Timestamp(meta.origin.max()))
        if not origins:
            continue
        latest = max(origins)
        publication = rows.loc[rows.origin.eq(latest)].sort_values(
            "interval_start", kind="stable"
        )
        if publication.interval_start.duplicated().any():
            raise HvacError("conflicting duplicate legacy interval identity")
        current_meta = (
            meta.loc[meta.origin.eq(latest)].iloc[-1]
            if len(meta.loc[meta.origin.eq(latest)])
            else None
        )
        flags = (
            publication.band_raised.to_numpy(dtype=float, na_value=np.nan)
            if live and "band_raised" in publication
            else (
                np.full(len(publication), np.nan)
                if live
                else _raised_flags(publication, current_meta)
            )
        )
        if not np.isin(flags[np.isfinite(flags)], [0.0, 1.0]).all():
            raise HvacError("invalid exact legacy band flags")
        lookup = {
            pd.Timestamp(stamp): i for i, stamp in enumerate(publication.interval_start)
        }
        for j, step in enumerate(grid):
            if selected[j] or blocked[j]:
                continue
            position = lookup.get(pd.Timestamp(step.start))
            if position is None:
                # A missing point inside the expected hourly coverage might be
                # archive loss. Do not substitute an older daily cell for it.
                # The legacy owner filters hourly publications to the next hour.
                if (
                    horizon == "hourly"
                    and latest <= step.start <= latest + pd.Timedelta(hours=1)
                ):
                    blocked[j] = True
                continue
            row = publication.iloc[position]
            duration = step.minutes / 60000
            values = np.array([row.lower_w, row.expected_w, row.upper_w], dtype=float)
            finite = np.isfinite(values) & (values >= 0)
            expected[j] = float(row.expected_w) * duration if finite[1] else np.nan
            if finite.all():
                quantiles[:, j] = values * duration
            if np.isfinite(flags[position]):
                value = float(row.upper_w if flags[position] else row.expected_w)
                if np.isfinite(value) and value >= 0:
                    planner[j] = value
            selected[j] = True
    diagnostics["legacy_quantile_intervals"] = int(
        np.isfinite(quantiles).all(axis=0).sum()
    )
    diagnostics["legacy_m3_intervals"] = int(np.isfinite(planner).sum())
    diagnostics["legacy_missing_or_ambiguous_band"] = int(
        (selected & ~np.isfinite(planner)).sum()
    )
    return BaselineForecast(expected, quantiles, planner, diagnostics)


def r1c_probabilities(
    view: BundleSnapshot, grid: tuple[GridInterval, ...], *, live: bool = False
) -> NDArray[np.float64]:
    """Read B5 on identical origin/step pairs, keeping null or absent values unknown.

    Args:
        view: One causal snapshot.
        grid: Five-minute first-hour comparison grid.
        live: Select actual archived R1c shadow probabilities.

    Returns:
        Probabilities, with NaN for unavailable evidence. No latest-other-origin
        substitution is allowed for the paired A1 comparison.

    Raises:
        ValueError: For duplicate identities or malformed persisted probabilities.
    """
    name = "r1c_shadow.parquet" if live else "r1c_replay.parquet"
    rows = view.baselines.get(name, pd.DataFrame())
    result = np.full(len(grid), np.nan)
    if rows.empty:
        return result
    rows = rows.loc[rows.origin.eq(view.origin)]
    start = "interval_start" if live else "step_start"
    value = "run_probability" if live else "p_run"
    if rows[start].duplicated().any():
        raise HvacError("duplicate R1c comparison identity")
    lookup = rows.set_index(start)[value]
    for i, step in enumerate(grid):
        observed = lookup.get(pd.Timestamp(step.start), np.nan)
        if pd.notna(observed):
            p = float(cast(float, observed))
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise HvacError("invalid R1c probability")
            result[i] = p
    return result
