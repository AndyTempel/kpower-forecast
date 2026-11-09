"""Grid-specific statistics computed from the complete in-memory ensemble."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
from numpy.typing import NDArray

from .types import GridInterval, HvacError, IntervalStats, MinutePaths, validate_grid


def _bounds(
    paths: MinutePaths, grid: tuple[GridInterval, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    validate_grid(grid)
    start = np.array([(step.start - paths.start).total_seconds() / 60 for step in grid])
    end = start + np.array([step.minutes for step in grid])
    if start[0] < 0 or end[-1] > paths.power_w.shape[1]:
        raise HvacError("statistics grid is outside the retained minute paths")
    return start, end


def _integral(
    values: np.ndarray, start: NDArray[np.float64], end: NDArray[np.float64]
) -> NDArray[np.float64]:
    cumulative = np.concatenate(
        [np.zeros((len(values), 1)), np.cumsum(values, axis=1, dtype=float)], axis=1
    )

    def at(offset: NDArray[np.float64]) -> NDArray[np.float64]:
        whole = np.floor(offset).astype(int)
        fraction = offset - whole
        safe = np.minimum(whole, values.shape[1] - 1)
        return np.asarray(
            cumulative[:, whole] + values[:, safe] * fraction, dtype=float
        )

    return at(end) - at(start)


def _touch(
    values: np.ndarray, start: NDArray[np.int64], end: NDArray[np.int64]
) -> NDArray[np.float64]:
    cumulative = np.concatenate(
        [
            np.zeros((len(values), 1), dtype=np.int64),
            np.cumsum(values, axis=1, dtype=np.int64),
        ],
        axis=1,
    )
    return np.asarray(
        np.mean(cumulative[:, end] - cumulative[:, start] > 0, axis=0), dtype=float
    )


def interval_stats(paths: MinutePaths, grid: tuple[GridInterval, ...]) -> IntervalStats:
    """Compute interval means, quantiles, run/start events and peaks on this grid.

    Args:
        paths: Every simulated minute path; display representatives are not accepted.
        grid: Contiguous intervals inside the path horizon.

    Returns:
        Read-only arrays. Quantiles and event probabilities are never combined
        from a finer-grid report. Fractional boundary minutes are weighted for
        energy and running fraction, with event probabilities based on overlap.

    Raises:
        ValueError: If the grid falls outside the retained paths.
    """
    start, end = _bounds(paths, grid)
    energy = _integral(paths.power_w, start, end) / 60000
    quantiles = np.quantile(
        energy, [0.05, 0.1, 0.5, 0.9, 0.95], axis=0, method="linear"
    )
    left = np.floor(start).astype(np.int64)
    right = np.ceil(end).astype(np.int64)
    peak = np.column_stack(
        [
            np.max(paths.power_w[:, a:b], axis=1)
            for a, b in zip(left, right, strict=True)
        ]
    )
    running = fraction = starts = None
    if paths.running is not None:
        running = _touch(paths.running, left, right)
        fraction = np.mean(_integral(paths.running, start, end), axis=0) / (end - start)
        if paths.starts is not None:
            starts = _touch(paths.starts, np.ceil(start).astype(np.int64), right)
    result = IntervalStats(
        grid,
        np.mean(energy, axis=0),
        quantiles[0],
        quantiles[1],
        quantiles[2],
        quantiles[3],
        quantiles[4],
        running,
        fraction,
        starts,
        np.quantile(peak, 0.9, axis=0, method="linear"),
    )
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def representatives(
    paths: MinutePaths, grid: tuple[GridInterval, ...], *, count: int = 16
) -> tuple[NDArray[np.float32], int]:
    """Select total-energy strata midpoints for display only, with no tail guarantee.

    Args:
        paths: Full ensemble.
        grid: Display grid.
        count: Representative count, not exceeding the path count.

    Returns:
        Read-only interval-mean watts and the closest representative to the
        ensemble median interval series in L1 distance.

    Raises:
        ValueError: If count or grid is invalid.
    """
    start, end = _bounds(paths, grid)
    if count < 1 or count > len(paths.power_w):
        raise HvacError("invalid representative count")
    order = np.argsort(paths.power_w.sum(axis=1, dtype=float), kind="stable")
    ranks = np.floor((np.arange(count) + 0.5) * len(order) / count).astype(int)
    energy = _integral(paths.power_w, start, end)
    mean_power = energy / (end - start)
    selected = np.asarray(mean_power[order[ranks]], dtype=np.float32)
    median = np.median(mean_power, axis=0)
    display = int(np.argmin(np.abs(selected - median).sum(axis=1)))
    selected.setflags(write=False)
    return selected, display
