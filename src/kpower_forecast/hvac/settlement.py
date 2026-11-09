"""Pure directional all-path settlement need, shared by planning and evaluation.

There is no recharge credit inside a block: this deliberately conservative
quantity sums each window's required discharge, independent of battery or price.
Component quantiles do not generally add to the total quantile.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from math import isfinite
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .types import HvacError, MinutePaths, utc


@dataclass(frozen=True, slots=True)
class SettlementWindow:
    """Original caller-supplied billing-window inputs, with signed ledger allowance."""

    start: datetime
    end: datetime
    block_id: str
    import_allowance_wh: float | None
    hard_import_limit_w: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", utc(self.start))
        object.__setattr__(self, "end", utc(self.end))
        if (
            self.end <= self.start
            or not isinstance(self.block_id, str)
            or not self.block_id
        ):
            raise HvacError("settlement window requires ordered bounds and a block id")
        for value in (self.import_allowance_wh, self.hard_import_limit_w):
            if value is not None and (isinstance(value, bool) or not isfinite(value)):
                raise HvacError("settlement numeric inputs must be finite or None")
        if self.hard_import_limit_w is not None and self.hard_import_limit_w < 0:
            raise HvacError("hard import limit cannot be negative")


@dataclass(frozen=True)
class BlockSettlement:
    """One maximal consecutive block, all quantiles reduced from full path needs."""

    block_id: str
    start: datetime
    end: datetime
    window_indices: tuple[int, ...]
    energy_q_wh: NDArray[np.float64]
    power_q_w: NDArray[np.float64]
    hard_energy_q_wh: NDArray[np.float64]
    average_energy_q_wh: NDArray[np.float64]
    paths_needing: float
    past_overrun_wh: float
    per_path_energy_wh: NDArray[np.float64]
    per_path_hard_wh: NDArray[np.float64]


@dataclass(frozen=True)
class SettlementNeed:
    """In-process result; original window inputs remain available for evidence."""

    quantiles: tuple[float, ...]
    windows: tuple[SettlementWindow, ...]
    blocks: tuple[BlockSettlement, ...]
    base_pv_resolution_min: int


def _water_level(
    import_w: NDArray[np.float64],
    weights: NDArray[np.float64],
    target_wmin: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve sum(min(import,lambda)*minutes)=target without exporting discharge."""
    order = np.argsort(import_w, axis=1)
    values = np.take_along_axis(import_w, order, axis=1)
    duration = np.broadcast_to(weights, import_w.shape)
    duration = np.take_along_axis(duration, order, axis=1)
    energy = values * duration
    before_energy = np.cumsum(energy, axis=1) - energy
    remaining_time = weights.sum() - (np.cumsum(duration, axis=1) - duration)
    candidate = (target_wmin[:, None] - before_energy) / remaining_time
    fits = candidate <= values
    first = np.argmax(fits, axis=1)
    level = candidate[np.arange(len(values)), first]
    level = np.where(fits.any(axis=1), level, values[:, -1])
    return np.clip(level, 0, values[:, -1])


def settlement_need(
    paths: MinutePaths,
    base_w: NDArray[np.floating],
    pv_w: NDArray[np.floating],
    windows: Sequence[SettlementWindow],
    quantiles: Sequence[float] = (0.5, 0.9),
    *,
    base_pv_resolution_min: int = 1,
) -> SettlementNeed:
    """Compute required discharge energy and feasible peak power on every path.

    Args:
        paths: Full in-memory minute ensemble, never its display representatives.
        base_w: Finite nonnegative base watts per minute, expanded by the caller.
        pv_w: Finite nonnegative PV watts on the same minute grid.
        windows: Ordered nonoverlapping original billing windows. The current
            window may begin before paths.start; only its future overlap is used.
            Its supplied allowance must already account for elapsed import.
        quantiles: Ordered distinct probabilities, reduced only after block sums.
        base_pv_resolution_min: Caller's piecewise-constant source resolution.

    Returns:
        Wh and W quantiles, full path need samples and past-overrun evidence.
        Export never refunds import. Hard shave happens first. A negative ledger
        allowance becomes zero future budget; its past overrun is separate.
        No recharge credit is applied within a block.

    Raises:
        ValueError: For malformed arrays, windows, bounds or quantiles. Missing
            actuals must be excluded by the caller, never filled by this function.
    """
    horizon = paths.power_w.shape[1]
    base = np.asarray(base_w, dtype=float)
    pv = np.asarray(pv_w, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if (
        base.shape != (horizon,)
        or pv.shape != (horizon,)
        or not np.isfinite(base).all()
        or not np.isfinite(pv).all()
        or (base < 0).any()
        or (pv < 0).any()
    ):
        raise HvacError("base/PV must be finite nonnegative minute arrays")
    if (
        q.ndim != 1
        or not len(q)
        or not np.isfinite(q).all()
        or (q < 0).any()
        or (q > 1).any()
        or (np.diff(q) <= 0).any()
    ):
        raise HvacError("quantiles must be increasing distinct probabilities")
    if (
        isinstance(base_pv_resolution_min, bool)
        or not isinstance(base_pv_resolution_min, int)
        or base_pv_resolution_min < 1
    ):
        raise HvacError("invalid base/PV source resolution")
    original = tuple(windows)
    if any(a.end > b.start for a, b in zip(original, original[1:], strict=False)):
        raise HvacError("settlement windows must be ordered and nonoverlapping")
    groups: list[list[int]] = []
    hard: list[NDArray[np.float64]] = []
    average: list[NDArray[np.float64]] = []
    peaks: list[NDArray[np.float64]] = []
    overruns: list[float] = []
    for i, window in enumerate(original):
        left = max((window.start - paths.start).total_seconds() / 60, 0.0)
        right = (window.end - paths.start).total_seconds() / 60
        if right <= left or right > horizon:
            raise HvacError("settlement window has no complete future path coverage")
        at = np.arange(int(np.floor(left)), int(np.ceil(right)))
        weights = np.minimum(at + 1, right) - np.maximum(at, left)
        net = paths.power_w[:, at].astype(float) + base[at] - pv[at]
        limit = window.hard_import_limit_w
        shaved = (
            np.maximum(net - limit, 0.0) if limit is not None else np.zeros_like(net)
        )
        importing = np.maximum(net - shaved, 0.0)
        hard_wh = np.sum(shaved * weights, axis=1) / 60
        imported_wh = np.sum(importing * weights, axis=1) / 60
        allowance = window.import_allowance_wh
        excess = (
            np.maximum(imported_wh - max(allowance, 0.0), 0.0)
            if allowance is not None
            else np.zeros(len(net))
        )
        level = _water_level(importing, weights, excess * 60)
        requirement = shaved + np.minimum(importing, level[:, None])
        hard.append(hard_wh)
        average.append(excess)
        peaks.append(np.max(requirement, axis=1))
        overruns.append(max(-allowance, 0.0) if allowance is not None else 0.0)
        if (
            i
            and original[i - 1].block_id == window.block_id
            and original[i - 1].end == window.start
        ):
            groups[-1].append(i)
        else:
            groups.append([i])
    blocks = []
    for indices in groups:
        h = np.sum([hard[i] for i in indices], axis=0)
        a = np.sum([average[i] for i in indices], axis=0)
        need = h + a
        power = np.column_stack([peaks[i] for i in indices])
        block = BlockSettlement(
            original[indices[0]].block_id,
            original[indices[0]].start,
            original[indices[-1]].end,
            tuple(indices),
            np.quantile(need, q),
            np.quantile(power, q, axis=0),
            np.quantile(h, q),
            np.quantile(a, q),
            float(np.mean(need > 0)),
            float(sum(overruns[i] for i in indices)),
            need,
            h,
        )
        for field in fields(block):
            value = getattr(block, field.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        blocks.append(block)
    return SettlementNeed(
        tuple(float(v) for v in q), original, tuple(blocks), base_pv_resolution_min
    )
