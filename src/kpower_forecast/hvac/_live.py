"""Live acceptance reads published statistics and reserves; it never runs a model."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd

from ._backtest import BacktestResult, _site_quality
from ._baselines import (
    BaselineForecast,
    legacy_baseline,
    r1c_probabilities,
    simple_baselines,
)
from ._scoring import _energy_windows
from .aggregate import _integral
from .bundle import BundleSnapshot, HvacEvaluationBundle
from .evaluation import actual_intervals, lead_bucket, pinball, regular_grid
from .settlement import SettlementWindow, settlement_need
from .types import GridInterval, HvacError, HvacModelConfig, MinutePaths, utc


def _official_energy(
    view: BundleSnapshot,
    rows: pd.DataFrame,
    baselines: dict[str, BaselineForecast],
) -> pd.DataFrame:
    """Use original actual-origin totals; never fill an unpublished prefix."""
    totals = view.outcomes.get("live_horizon_energy.parquet", pd.DataFrame())
    lookup: dict[int, float] = {}
    if not totals.empty:
        selected = totals.loc[totals.origin.eq(view.origin)].copy()
        if selected.duplicated(["horizon", "duration_h"]).any():
            raise HvacError("duplicate original horizon energy identity")
        if not selected.horizon.isin(["hourly", "daily"]).all():
            raise HvacError("unknown original horizon publication")
        selected["_priority"] = selected.horizon.map({"hourly": 0, "daily": 1})
        for row in selected.sort_values("_priority").itertuples():
            if isinstance(row.duration_h, bool) or isinstance(row.expected_kwh, bool):
                raise HvacError("invalid original horizon energy")
            duration = float(cast(float, row.duration_h))
            value = float(cast(float, row.expected_kwh))
            if duration not in (1, 6, 24) or not np.isfinite(value) or value < 0:
                raise HvacError("invalid original horizon energy")
            lookup.setdefault(int(duration), value)
    result = rows.copy()
    actual = view.outcomes["minutes.parquet"]
    additions: list[dict[str, object]] = []
    for duration in (1, 6, 24):
        end = view.origin + timedelta(hours=duration)
        start = pd.Timestamp(view.origin).floor("min")
        index = pd.date_range(
            start, pd.Timestamp(end).ceil("min"), freq="min", inclusive="left"
        )
        power = actual.reindex(index).power_w.to_numpy(dtype=float)
        known = np.isfinite(power)
        offset = (view.origin - start.to_pydatetime()).total_seconds() / 60
        left, right = np.array([offset]), np.array([offset + duration * 60])
        covered = float(_integral(known[None, :], left, right)[0, 0])
        actual_partial = float(
            _integral(np.where(known, power, 0.0)[None, :], left, right)[0, 0] / 60000
        )
        full = bool(np.isclose(covered, duration * 60, rtol=0, atol=1e-9))
        for model in ("cycle", *baselines):
            existing = result.index[
                result.interval_start.eq(view.origin)
                & result.duration_h.eq(duration)
                & result.model.eq(model)
            ]
            if len(existing) > 1:
                raise HvacError("duplicate official energy window")
            if len(existing) and (model != "cycle" or duration not in lookup):
                continue  # Aligned, complete published quarter means are additive.
            expected = (
                lookup.get(duration, np.nan)
                if model == "cycle"
                else (
                    float(baselines["B1"].expected_kwh[0] * 4 * duration)
                    if model == "B1"
                    else np.nan
                )
            )
            complete = bool(full and np.isfinite(expected))
            record: dict[str, object] = {
                "origin": view.origin,
                "interval_start": view.origin,
                "duration_h": duration,
                "lead_min": 0.0,
                "energy_scope": "first_horizon",
                "model": model,
                "eligible": complete,
                "coverage": covered / (duration * 60),
                "observed_minutes": covered,
                "actual_kwh": actual_partial if complete else np.nan,
                "expected_kwh": expected,
                "error_kwh": expected - actual_partial if complete else np.nan,
                "partial_actual_kwh": actual_partial if covered > 0 else np.nan,
                "partial_expected_kwh": (
                    expected
                    if complete
                    else (
                        expected * covered / (duration * 60)
                        if model == "B1"
                        else np.nan
                    )
                ),
                "partial_error_kwh": (
                    expected - actual_partial
                    if complete
                    else (
                        expected * covered / (duration * 60) - actual_partial
                        if model == "B1" and covered > 0
                        else np.nan
                    )
                ),
                "partial_method": (
                    "original_total_complete_window_only"
                    if model == "cycle"
                    else (
                        "causal_flat_mean"
                        if model == "B1"
                        else "missing_published_prefix"
                    )
                ),
                "horizon_evidence": (
                    "original_mean_total"
                    if model == "cycle" and duration in lookup
                    else (
                        "causal_flat_mean"
                        if model == "B1"
                        else "missing_published_prefix"
                    )
                ),
            }
            if len(existing):
                previous = result.loc[existing[0]]
                previous_mean = float(previous.expected_kwh)
                if np.isfinite(previous_mean) and not np.isclose(
                    previous_mean, expected, rtol=1e-9, atol=1e-9
                ):
                    raise HvacError(
                        "original horizon total contradicts complete published means"
                    )
                # Keep the separately labelled observed-footprint diagnostic
                # from published interval means when it can be computed.
                for key in (
                    "partial_actual_kwh",
                    "partial_expected_kwh",
                    "partial_error_kwh",
                    "partial_method",
                ):
                    record[key] = previous[key]
                result = result.drop(index=existing)
            additions.append(record)
    return pd.concat([result, pd.DataFrame.from_records(additions)], ignore_index=True)


def _original_number(rows: pd.DataFrame, name: str) -> float | None:
    if name not in rows:
        return None
    if rows[name].map(lambda value: isinstance(value, bool)).any():
        raise HvacError(f"invalid original publication {name}")
    values = pd.to_numeric(rows[name], errors="raise").dropna().unique()
    if len(values) > 1:
        raise HvacError(f"conflicting original publication {name}")
    if not len(values) or rows[name].isna().any():
        return None
    value = float(values[0])
    if (
        not np.isfinite(value)
        or value <= 0
        or (name == "n_paths" and (value < 16 or not value.is_integer()))
    ):
        raise HvacError(f"invalid original publication {name}")
    return value


def _published(
    rows: pd.DataFrame, origin: datetime, grid: tuple[GridInterval, ...]
) -> pd.DataFrame:
    selected = rows.loc[rows.origin.eq(origin)].copy()
    if selected.duplicated(["horizon", "interval_start"]).any():
        raise HvacError("duplicate published forecast identity")
    if not selected.horizon.isin(["hourly", "daily"]).all():
        raise HvacError("unknown published forecast horizon")
    for name in ("expected_w", "q10_w", "q50_w", "q90_w", "run_probability"):
        values = selected[name].dropna().to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (name == "run_probability" and (values > 1).any())
        ):
            raise HvacError("invalid published forecast statistic")
    if ((selected.q10_w > selected.q50_w) | (selected.q50_w > selected.q90_w)).any():
        raise HvacError("reversed published forecast quantiles")
    selected["_priority"] = selected.horizon.map({"hourly": 0, "daily": 1})
    selected = selected.sort_values("_priority").drop_duplicates(
        "interval_start", keep="first"
    )
    return selected.set_index("interval_start").reindex(
        pd.DatetimeIndex([step.start for step in grid])
    )


def _live_blocks(
    view: BundleSnapshot,
    baseline_grid: tuple[GridInterval, ...],
    legacy_power: np.ndarray,
) -> pd.DataFrame:
    rows = view.outcomes.get("live_reserves.parquet", pd.DataFrame())
    inputs = view.outcomes.get("live_window_inputs.parquet", pd.DataFrame())
    if rows.empty:
        return pd.DataFrame()
    rows = rows.loc[rows.forecast_origin.eq(view.origin) & rows["quantile"].eq(0.9)]
    keys = ["generated_at", "plan", "window_start", "window_end", "block_id"]
    if rows.duplicated(keys).any():
        raise HvacError("duplicate published q90 reserve identity")
    actual = view.outcomes["minutes.parquet"]
    site = view.outcomes.get("site_minutes.parquet", pd.DataFrame())
    validity = _site_quality(view, site) if not site.empty else pd.Series(dtype=bool)
    records: list[dict[str, object]] = []
    for row in rows.itertuples():
        start = cast(pd.Timestamp, row.window_start).to_pydatetime()
        end = cast(pd.Timestamp, row.window_end).to_pydatetime()
        path_start = max(start, pd.Timestamp(view.origin).floor("min").to_pydatetime())
        available = (
            inputs.loc[
                inputs.generated_at.eq(row.generated_at)
                & inputs.plan.eq(row.plan)
                & inputs.forecast_origin.eq(view.origin)
                & inputs.start.ge(start)
                & inputs.end.le(end)
            ].sort_values("start")
            if not inputs.empty
            else inputs
        )
        complete_inputs = bool(
            len(available)
            and available.start.iloc[0] == start
            and available.end.iloc[-1] == end
        )
        windows: list[SettlementWindow] = []
        if complete_inputs:
            for original in available.itertuples():
                windows.append(
                    SettlementWindow(
                        cast(pd.Timestamp, original.start).to_pydatetime(),
                        cast(pd.Timestamp, original.end).to_pydatetime(),
                        str(original.block_id),
                        (
                            float(cast(float, original.import_allowance_wh))
                            if pd.notna(original.import_allowance_wh)
                            else None
                        ),
                        (
                            float(cast(float, original.hard_import_limit_w))
                            if pd.notna(original.hard_import_limit_w)
                            else None
                        ),
                    )
                )
            complete_inputs = all(
                a.end == b.start for a, b in zip(windows, windows[1:], strict=False)
            ) and all(window.block_id == str(row.block_id) for window in windows)
        if end <= path_start:
            raise HvacError("published reserve ends before its forecast paths")
        index = pd.date_range(
            pd.Timestamp(path_start).floor("min"),
            pd.Timestamp(end).ceil("min"),
            freq="min",
            inclusive="left",
        )
        power = actual.reindex(index).power_w.to_numpy(dtype=float)
        base = (
            site.reindex(index)
            .get("base_w", pd.Series(np.nan, index=index))
            .to_numpy(dtype=float)
        )
        pv = (
            site.reindex(index)
            .get("pv_w", pd.Series(np.nan, index=index))
            .to_numpy(dtype=float)
        )
        known = bool(
            complete_inputs
            and np.isfinite(power).all()
            and np.isfinite(base).all()
            and np.isfinite(pv).all()
            and (base >= 0).all()
            and (pv >= 0).all()
            and validity.reindex(index, fill_value=False).all()
        )
        realized = hard_actual = np.nan
        if known:
            truth = settlement_need(
                MinutePaths(
                    index[0].to_pydatetime(), power[None, :].astype(np.float32), None
                ),
                base,
                pv,
                windows,
            ).blocks
            if len(truth) != 1:
                raise HvacError("original reserve inputs do not reconstruct one block")
            realized, hard_actual = (
                float(truth[0].per_path_energy_wh[0]),
                float(truth[0].per_path_hard_wh[0]),
            )
        legacy_minute_index = pd.date_range(
            baseline_grid[0].start, periods=len(legacy_power) * 15, freq="min"
        )
        legacy_path = (
            pd.Series(np.repeat(legacy_power, 15), index=legacy_minute_index)
            .reindex(index)
            .to_numpy(dtype=float)
        )
        legacy_need = legacy_hard = np.nan
        if known and np.isfinite(legacy_path).all():
            projected = settlement_need(
                MinutePaths(
                    index[0].to_pydatetime(),
                    legacy_path[None, :].astype(np.float32),
                    None,
                ),
                base,
                pv,
                windows,
            ).blocks[0]
            legacy_need, legacy_hard = (
                float(projected.energy_q_wh[1]),
                float(projected.hard_energy_q_wh[1]),
            )
        original_hard = (
            float(cast(float, row.hard_energy_wh))
            if pd.notna(row.hard_energy_wh)
            else np.nan
        )
        for model, predicted, hard in (
            ("cycle", float(cast(float, row.energy_wh)), original_hard),
            ("B3", legacy_need, legacy_hard),
        ):
            eligible = bool(known and np.isfinite(predicted))
            error, hard_error = realized - predicted, hard_actual - hard
            records.append(
                {
                    "origin": view.origin,
                    "generated_at": row.generated_at,
                    "plan": row.plan,
                    "block_start": start,
                    "block_end": end,
                    "block_id": str(row.block_id),
                    "model": model,
                    "lead_min": (start - view.origin).total_seconds() / 60,
                    "windows": len(windows),
                    "eligible": eligible,
                    "original_inputs_complete": complete_inputs,
                    "reserve_q90_wh": predicted,
                    "actual_need_wh": realized,
                    "q90_covered": float(realized <= predicted) if eligible else np.nan,
                    "pinball_q90": (
                        max(0.9 * error, -0.1 * error) if eligible else np.nan
                    ),
                    "realized_zero": float(realized == 0) if eligible else np.nan,
                    "hard_q90_wh": hard,
                    "hard_actual_wh": hard_actual,
                    "hard_pinball_q90": (
                        max(0.9 * hard_error, -0.1 * hard_error)
                        if eligible and np.isfinite(hard)
                        else np.nan
                    ),
                    "peak_crps_kwh": np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def live_score(
    bundle: HvacEvaluationBundle,
    config: HvacModelConfig,
    *,
    deploy: datetime,
    until: datetime | None = None,
) -> BacktestResult:
    """Score published origins in the original fourteen-day deployment window.

    Args:
        bundle: E1 live export, containing verified published versions and actuals.
        config: Bundle lag/timezone contract; current defaults never stand in for
            missing original n_paths or running guards.
        deploy: Deployment start, inclusive. A fix starts a new window.
        until: Optional exclusive origin cutoff, capped at deploy plus fourteen
            days. Outcome maturity follows the export's measured availability.

    Returns:
        Same aggregate report shape as backtest, with no fits or simulated paths.
        A discovery snapshot reads only the archive index from separated outcomes;
        each scored origin then uses its own single availability snapshot.

    Raises:
        ValueError: For malformed, contradictory or duplicate archived evidence.
    """
    deploy = utc(deploy)
    stop = min(
        utc(until) if until else deploy + timedelta(days=14),
        deploy + timedelta(days=14),
    )
    if stop <= deploy:
        raise HvacError("live scoring window must be positive")
    if (
        config.ingest_lag_min != bundle.manifest.ingest_lag_min
        or config.timezone != bundle.manifest.timezone
    ):
        raise HvacError("live config disagrees with bundle availability contract")
    discovery = bundle.available_at(deploy, include_inputs=True)
    publications = discovery.outcomes.get("live_forecasts.parquet", pd.DataFrame())
    if publications.empty:
        raise HvacError("no published live forecasts in bundle")
    origins = sorted(
        publications.loc[
            publications.origin.ge(deploy) & publications.origin.lt(stop), "origin"
        ].unique()
    )
    tables: dict[str, list[pd.DataFrame]] = {
        key: [] for key in ("quarters", "runs", "energy", "blocks", "origins")
    }
    for stamp in origins:
        origin = pd.Timestamp(stamp).to_pydatetime()
        view = (
            discovery
            if origin == deploy
            else bundle.available_at(origin, include_inputs=True)
        )
        published = view.outcomes["live_forecasts.parquet"]
        original = published.loc[published.origin.eq(origin)]
        future = original.loc[original.interval_start.ge(origin)]
        if future.empty:
            tables["origins"].append(
                pd.DataFrame(
                    [
                        {
                            "origin": origin,
                            "status": "skipped_no_future_published_intervals",
                        }
                    ]
                )
            )
            continue
        first = cast(pd.Timestamp, future.interval_start.min()).to_pydatetime()
        grid = regular_grid(first, minutes=15, hours=120)
        values = _published(published, origin, grid)
        size = _original_number(original, "n_paths")
        threshold = _original_number(original, "run_threshold_w")
        actual = view.outcomes["minutes.parquet"]
        baselines = simple_baselines(view, config, grid)
        baselines["B3"] = legacy_baseline(view, grid, live=True)
        quarter_grid = grid[: 72 * 4]
        observed = actual_intervals(actual, quarter_grid, threshold_w=threshold)
        distributions = {
            "cycle": (
                values.expected_w.to_numpy(dtype=float)[:288] / 4000,
                values[["q10_w", "q50_w", "q90_w"]].to_numpy(dtype=float).T[:, :288]
                / 4000,
            )
        }
        distributions.update(
            {
                name: (baseline.expected_kwh[:288], baseline.quantiles_kwh[:, :288])
                for name, baseline in baselines.items()
            }
        )
        quarter_rows: list[dict[str, object]] = []
        for name, (expected, bounds) in distributions.items():
            eligible = (
                observed.eligible_energy
                & np.isfinite(bounds).all(axis=0)
                & np.isfinite(expected)
            )
            score = np.full(len(quarter_grid), np.nan)
            score[eligible] = pinball(
                observed.energy_kwh[eligible], bounds[:, eligible]
            )
            for i, step in enumerate(quarter_grid):
                quarter_rows.append(
                    {
                        "origin": origin,
                        "interval_start": step.start,
                        "minutes": 15,
                        "lead_min": (step.start - origin).total_seconds() / 60,
                        "bucket": lead_bucket(
                            (step.start - origin).total_seconds() / 60
                        ),
                        "model": name,
                        "eligible": bool(eligible[i]),
                        "actual_kwh": observed.energy_kwh[i],
                        "expected_kwh": expected[i],
                        "q10_kwh": bounds[0, i],
                        "q50_kwh": bounds[1, i],
                        "q90_kwh": bounds[2, i],
                        "pinball": score[i],
                        "crps": np.nan,
                    }
                )
        tables["quarters"].append(pd.DataFrame.from_records(quarter_rows))
        run_rows: list[dict[str, object]] = []
        for minutes, hours in ((15, 72), (5, 2)):
            run_grid = regular_grid(first, minutes=minutes, hours=hours)
            outcomes = actual_intervals(actual, run_grid, threshold_w=threshold)
            if minutes == 15:
                probability = values.run_probability.to_numpy(dtype=float)[:288]
            else:
                near = view.outcomes.get("live_first_hour.parquet", pd.DataFrame())
                near = near.loc[near.origin.eq(origin)] if not near.empty else near
                if not near.empty and near.interval_start.duplicated().any():
                    raise HvacError("duplicate published first-two-hour identity")
                probability = (
                    near.set_index("interval_start")
                    .run_probability.reindex(
                        pd.DatetimeIndex([step.start for step in run_grid])
                    )
                    .to_numpy(dtype=float)
                    if not near.empty
                    else np.full(24, np.nan)
                )
            models = {"cycle": probability}
            if minutes == 5:
                models["B5"] = np.concatenate(
                    [
                        r1c_probabilities(view, run_grid[:12], live=True),
                        np.full(12, np.nan),
                    ]
                )
            for name, probabilities in models.items():
                if (
                    np.isfinite(probabilities)
                    & ((probabilities < 0) | (probabilities > 1))
                ).any():
                    raise HvacError("invalid published run probability")
                for i, step in enumerate(run_grid):
                    p = probabilities[i]
                    run_eligible = bool(outcomes.eligible_run[i] and np.isfinite(p))
                    lead = (step.start - origin).total_seconds() / 60
                    run_rows.append(
                        {
                            "origin": origin,
                            "interval_start": step.start,
                            "minutes": minutes,
                            "lead_min": lead,
                            "bucket": lead_bucket(lead),
                            "model": name,
                            "eligible": run_eligible,
                            "probability": p,
                            "actual_run": outcomes.ran[i],
                            "brier": (
                                (p - outcomes.ran[i]) ** 2 if run_eligible else np.nan
                            ),
                            "n_paths": size,
                        }
                    )
        tables["runs"].append(pd.DataFrame.from_records(run_rows))
        index = pd.date_range(first, periods=120 * 60, freq="min")
        predictions = {"cycle": np.repeat(values.expected_w.to_numpy(dtype=float), 15)}
        predictions.update(
            {
                name: np.repeat(baseline.expected_kwh * 4000, 15)
                for name, baseline in baselines.items()
            }
        )
        tables["energy"].append(
            _official_energy(
                view,
                _energy_windows(
                    origin,
                    index,
                    actual.reindex(index).power_w.to_numpy(dtype=float),
                    predictions,
                    cycle_method="published_quarter_mean_piecewise_constant",
                ),
                baselines,
            )
        )
        tables["blocks"].append(_live_blocks(view, grid, baselines["B3"].planner_w))
        tables["origins"].append(
            pd.DataFrame(
                [
                    {
                        "origin": origin,
                        "status": "scored_published",
                        "schedule": "published",
                        "n_paths": size,
                        "run_threshold_w": threshold,
                        "deploy": deploy,
                        "origin_stop": stop,
                        "exported_at": bundle.manifest.exported_at,
                        "verdict_matures_at": deploy + timedelta(days=17),
                    }
                ]
            )
        )
    return BacktestResult(
        {
            name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            for name, parts in tables.items()
        },
        config,
        "published",
        120,
    )
