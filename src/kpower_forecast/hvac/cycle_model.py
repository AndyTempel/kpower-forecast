"""Right-censored log-normal duration models for idle and run spells.

    log(idle) = b0 + b1*T + b2*sun + mode effects + b_r*log(previous run)
    log(run)  = c0 + c1*T + c2*sun + mode effects

T is the site outdoor temperature at the spell start. The previous run length
carries the heat just delivered into the idle that follows (longer run, longer
idle); it is the single strongest phase signal after elapsed time. Spells are
weighted by recency (half-life ``recency_half_life_days``), so slow seasonal
drift in the building is followed without hour-of-day terms that would memorise
past control. Physically, colder weather shortens idles and lengthens runs; a
fitted temperature slope of the wrong sign is dropped rather than extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Event

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._stats import censored_lognormal_aft, check_budget
from .types import HvacError, HvacModelConfig

MIN_COMPLETE_SPELLS: int = 20
MIN_MODE_SPELLS: int = 8
MIN_OUTDOOR_RANGE_C: float = 3.0
# Ridge penalties (log-duration units): radiation, mode effects, previous run.
SUN_RIDGE: float = 0.5
MODE_RIDGE: float = 2.0
PREV_RUN_RIDGE: float = 0.5


@dataclass(frozen=True)
class CycleFit:
    """Coefficients of both duration models; JSON-serializable."""

    modes: tuple[str, ...]
    idle_beta: tuple[float, ...]
    idle_sigma: float
    run_beta: tuple[float, ...]
    run_sigma: float
    typical_run_min: float
    outdoor_low_c: float
    outdoor_high_c: float
    complete_idle: int
    complete_run: int

    def __post_init__(self) -> None:
        width = 3 + len(self.modes)
        if (
            len(self.run_beta) != width
            or len(self.idle_beta) != width + 1
            or not np.isfinite([*self.idle_beta, *self.run_beta]).all()
            or not (0 < self.idle_sigma < 5 and 0 < self.run_sigma < 5)
            or not self.typical_run_min > 0
        ):
            raise HvacError("invalid cycle coefficients")

    def _design(
        self, outdoor_c: ArrayLike, mode: ArrayLike, sun: ArrayLike
    ) -> NDArray[np.float64]:
        t, m, s = np.broadcast_arrays(
            np.asarray(outdoor_c, dtype=float),
            np.asarray(mode, dtype=str),
            np.asarray(sun, dtype=float),
        )
        columns = [np.ones(t.shape), t, s] + [
            (m == label) * 1.0 for label in self.modes
        ]
        return np.stack(columns, axis=-1)

    def locations(
        self,
        outdoor_c: ArrayLike,
        mode: ArrayLike,
        sun: ArrayLike,
        prev_run_min: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Log-median idle and run durations (minutes).

        Args:
            outdoor_c: Site outdoor temperature.
            mode: Operating mode labels; unknown labels act as the reference mode.
            sun: Shortwave radiation in kW/m².
            prev_run_min: Length of the run preceding the idle.

        Returns:
            ``(mu_idle, mu_run)`` arrays of the broadcast shape.
        """
        x = self._design(outdoor_c, mode, sun)
        prev = np.log(np.maximum(np.asarray(prev_run_min, dtype=float), 1.0))
        mu_run = x @ np.asarray(self.run_beta)
        mu_idle = x @ np.asarray(self.idle_beta[:-1]) + self.idle_beta[-1] * prev
        return np.asarray(mu_idle), np.asarray(mu_run)

    def stationary(
        self, outdoor_c: float, mode: str, sun: float = 0.0
    ) -> tuple[float, float, float]:
        """Long-run duty and median idle/run at fixed conditions.

        The previous-run input is set to the median run itself (fixed point).

        Returns:
            ``(duty, idle_median_min, run_median_min)``.
        """
        _, mu_run = self.locations(outdoor_c, mode, sun, 1.0)
        run = float(np.exp(mu_run))
        mu_idle, _ = self.locations(outdoor_c, mode, sun, run)
        idle = float(np.exp(mu_idle))
        mean_run = run * np.exp(self.run_sigma**2 / 2)
        mean_idle = idle * np.exp(self.idle_sigma**2 / 2)
        return float(mean_run / (mean_run + mean_idle)), idle, run


def _fit_kind(
    rows: pd.DataFrame,
    modes: tuple[str, ...],
    *,
    idle: bool,
    weights: NDArray[np.float64],
    cancel: Event | None,
    deadline: float | None,
) -> tuple[NDArray[np.float64], float, bool]:
    """Fit one duration model; drop a wrong-sign temperature slope."""
    columns = [
        np.ones(len(rows)),
        rows.outdoor_c.to_numpy(dtype=float),
        rows.sun.to_numpy(dtype=float),
    ] + [(rows["mode"] == label).to_numpy(dtype=float) for label in modes]
    ridge = [0.0, 0.0, SUN_RIDGE] + [MODE_RIDGE] * len(modes)
    if idle:
        columns.append(np.log(np.maximum(rows.prev_run_min.to_numpy(dtype=float), 1.0)))
        ridge.append(PREV_RUN_RIDGE)
    x = np.column_stack(columns)
    y = rows.minutes.to_numpy(dtype=float)
    c = rows.censored.to_numpy(dtype=bool)
    fit = censored_lognormal_aft(
        x,
        y,
        c,
        ridge=np.asarray(ridge),
        weights=weights,
        cancel=cancel,
        deadline=deadline,
    )
    beta = np.array(fit.coefficients)
    sigma = fit.sigma
    # Idle must not lengthen as it gets colder, nor runs shorten.
    wrong_sign = beta[1] < 0 if idle else beta[1] > 0
    if wrong_sign:
        keep = [i for i in range(x.shape[1]) if i != 1]
        refit = censored_lognormal_aft(
            x[:, keep],
            y,
            c,
            ridge=np.asarray(ridge)[keep],
            weights=weights,
            cancel=cancel,
            deadline=deadline,
        )
        beta = np.insert(np.array(refit.coefficients), 1, 0.0)
        sigma = refit.sigma
        fit = refit
    if not fit.converged:
        raise HvacError("cycle_fit_not_converged")
    return beta, float(sigma), bool(wrong_sign)


def fit_cycles(
    sojourns: pd.DataFrame,
    cfg: HvacModelConfig,
    *,
    as_of: datetime,
    cancel: Event | None = None,
    deadline: float | None = None,
) -> tuple[CycleFit | None, str | None, dict[str, bool]]:
    """Fit idle and run duration models on recency-weighted spells.

    Args:
        sojourns: Output of :func:`extract_cycles`.
        cfg: Reference/off modes and recency half-life.
        as_of: Fit time; recency weights are measured from here.
        cancel: Caller cancellation flag.
        deadline: Absolute monotonic deadline.

    Returns:
        ``(fit, fallback_reason, flags)``; the fit is None with a reason when the
        history cannot identify cycles. Flags report dropped temperature slopes
        and modes without enough spells.
    """
    check_budget(cancel, deadline)
    rows = sojourns.loc[
        ~sojourns["mode"].isin(list(cfg.off_modes))
        & np.isfinite(sojourns.outdoor_c.to_numpy(dtype=float))
        & np.isfinite(sojourns.sun.to_numpy(dtype=float))
        & (sojourns.minutes > 0)
    ].copy()
    runs = rows.loc[rows.kind == "run"]
    idles = rows.loc[(rows.kind == "idle") & rows.prev_run_min.notna()]
    complete_runs = int((~runs.censored).sum())
    complete_idles = int((~idles.censored).sum())
    if min(complete_runs, complete_idles) < MIN_COMPLETE_SPELLS:
        return None, "insufficient_cycles", {}
    outdoor = rows.outdoor_c.to_numpy(dtype=float)
    if np.ptp(outdoor) < MIN_OUTDOOR_RANGE_C:
        return None, "narrow_outdoor_range", {}
    flags: dict[str, bool] = {}
    modes: list[str] = []
    for label in sorted(set(rows["mode"]) - {cfg.reference_mode}):
        complete = (
            (~rows.loc[rows["mode"] == label, "censored"])
            .groupby(rows.loc[rows["mode"] == label, "kind"])
            .sum()
        )
        if (
            complete.get("run", 0) >= MIN_MODE_SPELLS
            and complete.get("idle", 0) >= MIN_MODE_SPELLS
        ):
            modes.append(label)
        else:
            flags[f"mode_effect_unidentified:{label}"] = True
    labels = tuple(modes)

    def weights(frame: pd.DataFrame) -> NDArray[np.float64]:
        start_ns = pd.DatetimeIndex(frame.start).as_unit("ns").to_numpy("int64")
        age_days = (pd.Timestamp(as_of).as_unit("ns").value - start_ns) / 86400e9
        w = 0.5 ** (np.maximum(age_days, 0) / cfg.recency_half_life_days)
        return np.asarray(w / w.mean())

    try:
        idle_beta, idle_sigma, idle_flag = _fit_kind(
            idles,
            labels,
            idle=True,
            weights=weights(idles),
            cancel=cancel,
            deadline=deadline,
        )
        run_beta, run_sigma, run_flag = _fit_kind(
            runs,
            labels,
            idle=False,
            weights=weights(runs),
            cancel=cancel,
            deadline=deadline,
        )
    except HvacError as exc:
        # An unidentifiable duration model degrades to the energy-only tier.
        if str(exc) != "cycle_fit_not_converged":
            raise
        return None, "cycle_fit_not_converged", flags
    if idle_flag:
        flags["idle_temperature_slope_dropped"] = True
    if run_flag:
        flags["run_temperature_slope_dropped"] = True
    fit = CycleFit(
        labels,
        tuple(float(v) for v in idle_beta),
        idle_sigma * cfg.dispersion_inflation,
        tuple(float(v) for v in run_beta),
        run_sigma * cfg.dispersion_inflation,
        float(runs.loc[~runs.censored, "minutes"].median()),
        float(np.quantile(outdoor, 0.02)),
        float(np.quantile(outdoor, 0.98)),
        complete_idles,
        complete_runs,
    )
    return fit, None, flags
