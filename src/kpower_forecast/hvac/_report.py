"""Matched-pair metric summaries and pre-registered gate decisions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import JsonValue

from .evaluation import LEAD_BUCKETS, calibration_slope, reliability_bins


def _pairs(
    rows: pd.DataFrame, baseline: str, keys: list[str], value: str
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    eligible = rows.loc[rows.eligible]
    cycle = eligible.loc[eligible.model.eq("cycle"), keys + [value]]
    other = eligible.loc[eligible.model.eq(baseline), keys + [value]]
    if cycle.duplicated(keys).any() or other.duplicated(keys).any():
        raise ValueError("duplicate score identity")
    return cycle.merge(
        other,
        on=keys,
        how="inner",
        suffixes=("_cycle", "_baseline"),
        validate="one_to_one",
    )


def _decision(checks: Sequence[object]) -> str:
    if any(value is not None and not isinstance(value, bool) for value in checks):
        raise ValueError("gate subcriteria must be boolean or unknown")
    if any(value is False for value in checks):
        return "fail"
    return (
        "unscorable" if not checks or any(value is None for value in checks) else "pass"
    )


def summarize(
    tables: Mapping[str, pd.DataFrame],
    *,
    n_paths: int,
    hard_gates: Mapping[str, bool] | None = None,
) -> dict[str, JsonValue]:
    """Summarize eligible evidence and evaluate fixed gates without inventing passes.

    Args:
        tables: Per-origin score tables from development or archived live scoring.
        n_paths: Original ensemble size, used only for A5 logit boundary clipping.
        hard_gates: Independently verified A6/A7 results. Missing proof stays
            unscorable; host runtime is never substituted for the actual Pi gate.

    Returns:
        JSON-safe summaries, matched counts, calibration and A1–A7 decisions.
        Missing evidence cannot satisfy a gate. Known failed subcriteria remain
        failures even if another subcriterion is unscorable.

    Raises:
        ValueError: For duplicate comparison identities or invalid probabilities.
    """
    summaries: dict[str, object] = {}
    quarters = tables.get("quarters", pd.DataFrame())
    runs = tables.get("runs", pd.DataFrame())
    energy = tables.get("energy", pd.DataFrame())
    blocks = tables.get("blocks", pd.DataFrame())
    for name, rows, group, columns in (
        ("M1", quarters, ["model", "bucket"], ["pinball", "crps"]),
        ("M2", runs, ["model", "minutes", "bucket"], ["brier"]),
        (
            "M3",
            blocks,
            ["model"],
            [
                "pinball_q90",
                "q90_covered",
                "realized_zero",
                "hard_pinball_q90",
                "peak_crps_kwh",
            ],
        ),
    ):
        if rows.empty:
            summaries[name] = []
            continue
        result = []
        for identity, part in rows.groupby(group, dropna=False):
            keys = identity if isinstance(identity, tuple) else (identity,)
            record = dict(zip(group, keys, strict=True))
            record.update(
                {"attempted": len(part), "eligible": int(part.eligible.sum())}
            )
            record.update({column: float(part[column].mean()) for column in columns})
            result.append(record)
        summaries[name] = result
    energy_summary = []
    if not energy.empty:
        for (model, duration), part in energy.groupby(["model", "duration_h"]):
            official = part.loc[part.eligible]
            energy_summary.append(
                {
                    "model": model,
                    "duration_h": duration,
                    "attempted": len(part),
                    "eligible": len(official),
                    "mae_kwh": float(official.error_kwh.abs().mean()),
                    "bias_kwh": float(official.error_kwh.mean()),
                    "actual_mean_kwh": float(official.actual_kwh.mean()),
                    "partial_n": int(part.partial_error_kwh.notna().sum()),
                    "partial_mae_kwh": float(part.partial_error_kwh.abs().mean()),
                    "partial_bias_kwh": float(part.partial_error_kwh.mean()),
                    "mean_coverage": float(part.coverage.mean()),
                }
            )
    summaries["M6"] = energy_summary
    gates: dict[str, object] = {}
    a1 = []
    for bucket, _, _ in LEAD_BUCKETS[:3]:
        source = (
            runs.loc[runs.minutes.eq(5) & runs.bucket.eq(bucket)]
            if not runs.empty
            else runs
        )
        pairs = _pairs(source, "B5", ["origin", "interval_start", "minutes"], "brier")
        n = len(pairs)
        cycle = float(pairs.brier_cycle.mean()) if n else None
        reference = float(pairs.brier_baseline.mean()) if n else None
        a1.append(
            {
                "bucket": bucket,
                "n": n,
                "cycle": cycle,
                "B5": reference,
                "pass": (
                    bool(cycle <= reference + 0.005)
                    if cycle is not None and reference is not None
                    else None
                ),
            }
        )
    gates["A1"] = {"status": _decision([row["pass"] for row in a1]), "checks": a1}
    a2 = []
    for i, (bucket, _, _) in enumerate(LEAD_BUCKETS):
        for baseline in ("B1", "B2", "B3") if i >= 3 else ("B1", "B2"):
            source = (
                quarters.loc[quarters.bucket.eq(bucket)]
                if not quarters.empty
                else quarters
            )
            pairs = _pairs(source, baseline, ["origin", "interval_start"], "pinball")
            n = len(pairs)
            cycle = float(pairs.pinball_cycle.mean()) if n else None
            reference = float(pairs.pinball_baseline.mean()) if n else None
            passed = None
            if cycle is not None and reference is not None:
                passed = (
                    bool(cycle <= 0.85 * reference)
                    if baseline == "B3"
                    else bool(cycle < reference)
                )
            a2.append(
                {
                    "bucket": bucket,
                    "baseline": baseline,
                    "n": n,
                    "cycle": cycle,
                    "reference": reference,
                    "pass": passed,
                }
            )
    gates["A2"] = {"status": _decision([row["pass"] for row in a2]), "checks": a2}
    a3 = []
    for name, left, right in (("1-24h", 60, 1440), ("24-72h", 1440, 4320)):
        source = (
            blocks.loc[(blocks.lead_min >= left) & (blocks.lead_min < right)]
            if not blocks.empty
            else blocks
        )
        eligible = (
            source.loc[source.eligible & source.model.eq("cycle")]
            if not source.empty
            else source
        )
        pairs = _pairs(
            source,
            "B3",
            ["origin", "block_start", "block_end", "block_id"],
            "pinball_q90",
        )
        coverage = float(eligible.q90_covered.mean()) if len(eligible) else None
        cycle = float(pairs.pinball_q90_cycle.mean()) if len(pairs) else None
        reference = float(pairs.pinball_q90_baseline.mean()) if len(pairs) else None
        a3.append(
            {
                "bucket": name,
                "coverage_n": len(eligible),
                "paired_n": len(pairs),
                "coverage": coverage,
                "cycle_pinball": cycle,
                "B3_pinball": reference,
                "coverage_pass": (
                    bool(0.85 <= coverage <= 0.97) if coverage is not None else None
                ),
                "pinball_pass": (
                    bool(cycle <= 0.8 * reference)
                    if cycle is not None and reference is not None
                    else None
                ),
            }
        )
    gates["A3"] = {
        "status": _decision(
            [row[key] for row in a3 for key in ("coverage_pass", "pinball_pass")]
        ),
        "checks": a3,
    }
    day = energy.loc[energy.duration_h.eq(24)] if not energy.empty else energy
    pairs = _pairs(day, "B1", ["origin", "interval_start", "duration_h"], "error_kwh")
    a4: dict[str, object] = {"paired_n": len(pairs), "status": "unscorable"}
    if len(pairs):
        actual = day.loc[
            day.eligible & day.model.eq("cycle"),
            ["origin", "interval_start", "duration_h", "actual_kwh"],
        ]
        matched = pairs.merge(
            actual, on=["origin", "interval_start", "duration_h"], validate="one_to_one"
        )
        mae, baseline_mae = (
            float(matched.error_kwh_cycle.abs().mean()),
            float(matched.error_kwh_baseline.abs().mean()),
        )
        bias, actual_mean = (
            float(matched.error_kwh_cycle.mean()),
            float(matched.actual_kwh.mean()),
        )
        a4.update(
            {
                "mae_kwh": mae,
                "B1_mae_kwh": baseline_mae,
                "bias_kwh": bias,
                "actual_mean_kwh": actual_mean,
                "status": _decision(
                    [mae <= baseline_mae, abs(bias) <= 0.05 * actual_mean]
                ),
            }
        )
    gates["A4"] = a4
    calibrated = (
        runs.loc[
            runs.model.eq("cycle")
            & runs.minutes.eq(15)
            & runs.lead_min.ge(60)
            & runs.eligible
        ]
        if not runs.empty
        else runs
    )
    calibration_sizes: int | np.ndarray = n_paths
    excluded_size = 0
    if len(calibrated) and "n_paths" in calibrated:
        sizes = pd.to_numeric(calibrated.n_paths, errors="coerce")
        known_size = sizes.notna() & sizes.ge(2) & sizes.eq(np.floor(sizes))
        excluded_size = int((~known_size).sum())
        calibrated = calibrated.loc[known_size]
        calibration_sizes = sizes.loc[known_size].to_numpy(dtype=np.int64)
    fit = (
        calibration_slope(
            calibrated.probability.to_numpy(dtype=float),
            calibrated.actual_run.to_numpy(dtype=float),
            n_paths=calibration_sizes,
        )
        if len(calibrated)
        else None
    )
    gates["A5"] = {
        "status": _decision(
            [
                (
                    bool(0.8 <= fit.slope <= 1.2)
                    if fit is not None and fit.slope is not None
                    else None
                )
            ]
        ),
        "calibration": (
            asdict(fit)
            if fit
            else {"reason": "no_eligible_run_pairs", "observations": 0}
        ),
        "excluded_unknown_ensemble_size": excluded_size,
    }
    for key in ("A6", "A7"):
        value = (hard_gates or {}).get(key)
        gates[key] = {
            "status": _decision([value]),
            "hard": True,
            "evidence": "supplied_separately" if value is not None else "not_supplied",
        }
    summaries["gates"] = gates
    curves = []
    if not runs.empty:
        for (curve_model, minutes, curve_bucket), part in runs.loc[
            runs.eligible
        ].groupby(["model", "minutes", "bucket"]):
            curves.append(
                {
                    "model": curve_model,
                    "minutes": minutes,
                    "bucket": curve_bucket,
                    "bins": reliability_bins(
                        part.probability.to_numpy(dtype=float),
                        part.actual_run.to_numpy(dtype=float),
                    ),
                }
            )
    summaries["reliability"] = curves
    cadence = tables.get("cadence", pd.DataFrame())
    summaries["M5"] = (
        []
        if cadence.empty
        else [
            {
                "duration_h": duration,
                "attempted": len(part),
                "eligible": int(part.eligible.sum()),
                "start_count_crps": float(part.start_count_crps.mean()),
                "run_length_ks": float(part.run_length_ks.mean()),
                "idle_length_ks": float(part.idle_length_ks.mean()),
            }
            for duration, part in cadence.groupby("duration_h")
        ]
    )
    peak_summary: dict[str, object] = {"eligible": 0, "peak_crps_kwh": None}
    if not blocks.empty:
        peak_summary = {
            "eligible": int(blocks.peak_crps_kwh.notna().sum()),
            "peak_crps_kwh": float(blocks.peak_crps_kwh.mean()),
        }
    summaries["M4"] = peak_summary
    runtime = tables.get("runtime", pd.DataFrame())
    summaries["M7"] = (
        {
            "origins": len(runtime),
            "fit_max_s": float(runtime.fit_seconds.max()),
            "predict_max_s": float(runtime.predict_seconds.max()),
            "predict_p95_s": float(runtime.predict_seconds.quantile(0.95)),
            "peak_added_rss_mb": float(runtime.peak_added_rss_mb.max()),
            "scope": "evaluation_host_not_pi_gate",
        }
        if not runtime.empty
        else {"origins": 0}
    )
    comfort = []
    if not quarters.empty and "future_comfort_off" in quarters:
        for (comfort_model, affected), part in quarters.groupby(
            ["model", "future_comfort_off"]
        ):
            comfort.append(
                {
                    "model": comfort_model,
                    "future_comfort_off": affected,
                    "attempted": len(part),
                    "eligible": int(part.eligible.sum()),
                    "pinball": float(part.pinball.mean()),
                }
            )
    summaries["future_comfort_off_error"] = comfort
    # Convert pandas/NumPy scalar types and unknown floating results into strict
    # JSON values. Serialization never includes the full minute ensemble.
    import json

    return cast_json(json.loads(pd.Series(summaries).to_json()))


def cast_json(value: object) -> dict[str, JsonValue]:
    """Validate a report JSON object after pandas normalizes unknown numbers."""
    from pydantic import TypeAdapter

    return TypeAdapter(dict[str, JsonValue]).validate_python(value)
