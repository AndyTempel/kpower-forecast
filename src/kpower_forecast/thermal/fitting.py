"""Shared bounded RC fit search for runtime and offline diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitCandidate:
    """One diagnostic optimum; rejected coefficients remain diagnostic only."""

    tau_hours: float
    gain_c_per_kw: float
    offset_c: float
    loss_c2: float
    condition_number: float
    boundary: str | None
    profile_relative_range: float

    @property
    def parameters(self) -> tuple[float, float, float]:
        """Return the coefficients in the model's canonical order."""
        return self.tau_hours, self.gain_c_per_kw, self.offset_c


def fit_candidate(
    starts: np.ndarray,
    ends: np.ndarray,
    outdoors: np.ndarray,
    power_kw: np.ndarray,
    hours: np.ndarray,
    *,
    min_tau: float,
    max_tau: float,
    grid_points: int,
    weights: np.ndarray | None = None,
) -> FitCandidate | None:
    """Search the existing RC grid, optionally with offline confidence weights.

    Args:
        starts: Genuine indoor start readings.
        ends: Genuine indoor end readings.
        outdoors: Covered mean outdoor forcing.
        power_kw: Attributable mean HVAC electric input.
        hours: Exact elapsed observation times.
        min_tau: Lower physical time-constant limit.
        max_tau: Upper physical time-constant limit.
        grid_points: Number of logarithmically spaced candidates.
        weights: Optional finite positive row weights; runtime omits them.

    Returns:
        Best unconstrained coefficients on the bounded grid, or None if singular.

    Raises:
        ValueError: If weights are malformed.
    """
    if weights is not None and (
        weights.shape != starts.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("fit weights must be finite, positive and aligned")
    best: tuple[float, float, float, float, float] | None = None
    losses: list[float] = []
    for tau in np.geomspace(min_tau, max_tau, grid_points):
        decay = -np.expm1(-hours / tau)
        target = ends - (1 - decay) * starts - decay * outdoors
        design = np.column_stack((decay * power_kw, decay))
        solve_design, solve_target = design, target
        if weights is not None:
            root = np.sqrt(weights)
            solve_design, solve_target = design * root[:, None], target * root
        if np.linalg.matrix_rank(solve_design) < 2:
            continue
        gain, offset = np.linalg.lstsq(solve_design, solve_target, rcond=None)[0]
        residual = target - design @ np.array([gain, offset])
        loss = float(
            np.mean(residual**2)
            if weights is None
            else np.average(residual**2, weights=weights)
        )
        if not np.isfinite(loss):
            continue
        losses.append(loss)
        if best is None or loss < best[3]:
            best = (
                float(tau),
                float(gain),
                float(offset),
                loss,
                float(np.linalg.cond(solve_design)),
            )
    if best is None:
        return None
    boundary = (
        "lower" if best[0] == min_tau else "upper" if best[0] == max_tau else None
    )
    return FitCandidate(
        *best,
        boundary=boundary,
        profile_relative_range=(max(losses) - min(losses)) / max(max(losses), 1e-12),
    )
