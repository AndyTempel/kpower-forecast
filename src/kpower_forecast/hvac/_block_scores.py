"""Block scores using the shared settlement function and all retained paths."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._baselines import BaselineForecast
from ._stats import sample_crps
from .aggregate import _integral
from .settlement import SettlementWindow, settlement_need
from .types import HvacError, HvacForecast, MinutePaths


def block_scores(
    forecast: HvacForecast,
    actual: pd.DataFrame,
    site: pd.DataFrame,
    site_valid: pd.Series,
    windows: Sequence[SettlementWindow],
    baselines: Mapping[str, BaselineForecast],
) -> pd.DataFrame:
    """Compute M3/M3h reserve and M4 quarter-peak scores per consecutive block.

    Args:
        forecast: Full ensemble, with minute- and quarter-aligned offline origin.
        actual: Qualified HVAC outcomes.
        site: Actual minute base and PV, without imputed gaps.
        site_valid: Caller-resolved owning attribution-quality mask. Observation
            alone is insufficient; unknown evidence must be False.
        windows: Original allowances and limits. Offline callers use future
            windows; live callers must retain the archived original ledger basis.
        baselines: Quarter planner powers, with approved development B3 proxy or
            exact live flags. Missing evidence excludes the whole block.

    Returns:
        Explicit block eligibility, q90 coverage/pinball, hard part and peak CRPS.
        Missing minutes exclude a whole block rather than shrinking its identity.

    Raises:
        ValueError: For unordered or non-quarter windows, incompatible masks or
            path bounds. The pure settlement function validates numeric inputs.
    """
    if forecast.origin != forecast.minute_paths.start:
        raise HvacError("offline block scores require origin-aligned minute paths")
    if site_valid.index.has_duplicates or site_valid.isna().any():
        raise HvacError("site quality mask must be explicit and uniquely indexed")
    groups: list[list[SettlementWindow]] = []
    for window in windows:
        if (
            window.start.second
            or window.start.microsecond
            or window.end - window.start != pd.Timedelta(minutes=15)
            or window.start < forecast.origin
            or window.end > forecast.minute_paths.end
        ):
            raise HvacError("offline block windows must be complete future quarters")
        if groups and window.start < groups[-1][-1].end:
            raise HvacError("unordered block windows")
        if (
            groups
            and window.block_id == groups[-1][-1].block_id
            and window.start == groups[-1][-1].end
        ):
            groups[-1].append(window)
        else:
            groups.append([window])
    records: list[dict[str, object]] = []
    for group in groups:
        start, end = group[0].start, group[-1].end
        left = int((start - forecast.origin).total_seconds() // 60)
        right = int((end - forecast.origin).total_seconds() // 60)
        index = pd.date_range(start, end, freq="min", inclusive="left")
        observed = actual.reindex(index).power_w.to_numpy(dtype=float)
        actual_known = bool(np.isfinite(observed).all())
        source = site.reindex(index)
        base = source.get("base_w", pd.Series(np.nan, index=index)).to_numpy(
            dtype=float
        )
        pv = source.get("pv_w", pd.Series(np.nan, index=index)).to_numpy(dtype=float)
        valid = site_valid.reindex(index, fill_value=False).to_numpy(dtype=bool)
        site_known = bool(
            valid.all()
            and np.isfinite(base).all()
            and np.isfinite(pv).all()
            and (base >= 0).all()
            and (pv >= 0).all()
        )
        samples = np.asarray(
            forecast.minute_paths.power_w[:, left:right], dtype=np.float32
        )
        models: dict[str, NDArray[np.float32]] = {"cycle": samples}
        for name, baseline in baselines.items():
            power = np.repeat(baseline.planner_w, 15)
            if len(power) < right:
                raise HvacError("block baseline does not cover its grid")
            models[name] = np.asarray(power[None, left:right], dtype=np.float32)
        truth = None
        if actual_known and site_known:
            truth = settlement_need(
                MinutePaths(start, observed[None, :].astype(np.float32), None),
                base,
                pv,
                group,
            ).blocks[0]
        for name, power_paths in models.items():
            eligible = bool(truth is not None and np.isfinite(power_paths).all())
            q90 = realized = score = hard_q90 = hard_actual = hard_score = coverage = (
                np.nan
            )
            if eligible:
                assert truth is not None
                predicted = settlement_need(
                    MinutePaths(start, power_paths, None), base, pv, group
                ).blocks[0]
                q90 = float(predicted.energy_q_wh[1])
                realized = float(truth.per_path_energy_wh[0])
                error = realized - q90
                score = max(0.9 * error, -0.1 * error)
                coverage = float(realized <= q90)
                hard_q90 = float(predicted.hard_energy_q_wh[1])
                hard_actual = float(truth.per_path_hard_wh[0])
                hard_error = hard_actual - hard_q90
                hard_score = max(0.9 * hard_error, -0.1 * hard_error)
            peak_crps = np.nan
            if name == "cycle" and actual_known:
                left_q = np.arange(0, len(index), 15, dtype=float)
                path_quarters = _integral(power_paths, left_q, left_q + 15) / 60000
                actual_quarters = observed.reshape(-1, 15).sum(axis=1) / 60000
                peak_crps = float(
                    sample_crps(
                        np.max(path_quarters, axis=1)[:, None],
                        [float(actual_quarters.max())],
                    )[0]
                )
            records.append(
                {
                    "origin": forecast.origin,
                    "block_start": start,
                    "block_end": end,
                    "block_id": group[0].block_id,
                    "model": name,
                    "lead_min": (start - forecast.origin).total_seconds() / 60,
                    "windows": len(group),
                    "eligible": eligible,
                    "actual_complete": actual_known,
                    "site_complete": site_known,
                    "reserve_q90_wh": q90,
                    "actual_need_wh": realized,
                    "q90_covered": coverage,
                    "pinball_q90": score,
                    "realized_zero": float(realized == 0) if eligible else np.nan,
                    "hard_q90_wh": hard_q90,
                    "hard_actual_wh": hard_actual,
                    "hard_pinball_q90": hard_score,
                    "peak_crps_kwh": peak_crps,
                }
            )
    return pd.DataFrame.from_records(records)
