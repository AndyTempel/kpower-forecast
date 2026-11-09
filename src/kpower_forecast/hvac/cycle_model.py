"""Physics-shaped, right-censored cycle duration models and day bootstrap fits."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._stats import censored_lognormal_aft, check_budget, normal_cdf
from .types import HvacCapabilities, HvacError, HvacModelConfig

OVERLOAD_MARGIN: float = 0.05


@dataclass(frozen=True)
class CycleParameters:
    """One fitted parameter set, with fixed physical shape outside training range."""

    t_bal_c: float
    idle_modes: tuple[str, ...]
    idle_beta: tuple[float, ...]
    sigma_idle: float
    rho: float
    capacity_slope: float
    run_modes: tuple[str, ...]
    run_beta: tuple[float, ...]
    sigma_run: float
    recovery_multiplier: float = 1.0

    def rates(
        self,
        outdoor_c: ArrayLike,
        mode: ArrayLike,
        sun: ArrayLike = 0.0,
        hour_sin: ArrayLike = 0.0,
        hour_cos: ArrayLike = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Evaluate idle rate, total rate and overload without a run-rate floor.

        Args:
            outdoor_c: Site outdoor temperature per path/event.
            mode: Current caller mode categories.
            sun: Shortwave radiation in kW/m².
            hour_sin: Local time-of-day sine.
            hour_cos: Local time-of-day cosine.

        Returns:
            Arrays (a, r, overloaded); a and b describe log-normal medians.
        """
        t, m, s, hs, hc = np.broadcast_arrays(
            np.asarray(outdoor_c, dtype=float),
            np.asarray(mode, dtype=str),
            np.asarray(sun, dtype=float),
            np.asarray(hour_sin, dtype=float),
            np.asarray(hour_cos, dtype=float),
        )
        beta = np.asarray(self.idle_beta)
        location = beta[0] + sum(
            (beta[1 + i] * (m == label) for i, label in enumerate(self.idle_modes)),
            start=np.zeros(t.shape),
        )
        location = location + beta[-3] * s + beta[-2] * hs + beta[-1] * hc
        a = np.maximum(self.t_bal_c - t, 0) * np.exp(-location)
        r = self.rho * np.clip(1 + self.capacity_slope * (t - 7), 0.4, 1.3)
        return np.asarray(a), np.asarray(r), np.asarray(a >= (1 - OVERLOAD_MARGIN) * r)

    def locations(
        self,
        outdoor_c: ArrayLike,
        mode: ArrayLike,
        sun: ArrayLike = 0.0,
        hour_sin: ArrayLike = 0.0,
        hour_cos: ArrayLike = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Return log-median idle/run duration, with infinite run under overload.

        Args:
            outdoor_c: Outdoor temperature.
            mode: Mode categories.
            sun: Radiation /1000.
            hour_sin: Local hour sine.
            hour_cos: Local hour cosine.

        Returns:
            mu_idle, mu_run and overloaded arrays. Zero demand gives infinite idle.
        """
        a, r, over = self.rates(outdoor_c, mode, sun, hour_sin, hour_cos)
        m = np.broadcast_to(np.asarray(mode, dtype=str), a.shape)
        effect = sum(
            (
                b * (m == label)
                for label, b in zip(self.run_modes, self.run_beta, strict=True)
            ),
            start=np.zeros(a.shape),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_idle = -np.log(a)
            mu_run = np.where(
                over, np.inf, -np.log(np.where(over, 1.0, r - a)) + effect
            )
        return np.asarray(mu_idle), np.asarray(mu_run), over

    def stationary(
        self,
        outdoor_c: ArrayLike,
        mode: ArrayLike,
        *,
        interval_min: float = 15.0,
        sun: ArrayLike = 0.0,
        hour_sin: ArrayLike = 0.0,
        hour_cos: ArrayLike = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute mean duty and interval-touch probability, not median duty.

        Args:
            outdoor_c: Outdoor temperatures.
            mode: Mode categories.
            interval_min: Positive interval duration.
            sun: Radiation /1000.
            hour_sin: Local hour sine.
            hour_cos: Local hour cosine.

        Returns:
            Stationary run fraction and run-touch probability.
        """
        if interval_min <= 0:
            raise HvacError("touch interval must be positive")
        mi, mr, over = self.locations(outdoor_c, mode, sun, hour_sin, hour_cos)
        # For overload evaluate harmless finite stand-ins before choosing duty=1.
        ei = np.exp(np.where(over, 0.0, mi) + self.sigma_idle**2 / 2)
        er = np.exp(np.where(over, 0.0, mr) + self.sigma_run**2 / 2)
        si = self.sigma_idle
        z = (np.log(interval_min) - mi) / si
        partial = np.exp(np.minimum(mi + si**2 / 2, 700)) * normal_cdf(
            z - si
        ) + interval_min * normal_cdf(-z)
        partial = np.where(np.isposinf(mi), interval_min, partial)
        duty = np.where(over, 1.0, er / (er + ei))
        touch = np.where(over, 1.0, (er + partial) / (er + ei))
        return np.asarray(duty), np.asarray(touch)


@dataclass(frozen=True)
class SwitchEffect:
    """Censored outcome distribution for an observed mode-switch context."""

    from_mode: str
    to_mode: str
    state: str
    mu: float
    sigma: float
    observations: int


@dataclass(frozen=True)
class CycleFit:
    """Full fit and bootstrap sets with profile/convergence evidence."""

    parameter_sets: tuple[CycleParameters, ...]
    t_bal_profile: tuple[tuple[float, float | None], ...]
    t_bal_at_grid_edge: bool
    overload_rows_in_fit: int
    unidentified_modes: tuple[str, ...]
    defrost_rate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    defrost_length_min: float = 0.0
    switch_effects: tuple[SwitchEffect, ...] = ()
    bootstrap_grid_edges: int = 0
    defrost_hourly_rate: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _design(
    rows: pd.DataFrame, modes: tuple[str, ...], *, idle: bool
) -> NDArray[np.float64]:
    blocks = [np.ones(len(rows))] if idle else []
    blocks.extend(np.asarray(rows["mode"].eq(mode), dtype=float) for mode in modes)
    if idle:
        blocks.extend(
            rows[c].to_numpy(dtype=float) for c in ("sun", "hour_sin", "hour_cos")
        )
    return np.column_stack(blocks) if blocks else np.empty((len(rows), 0))


def _supported_modes(rows: pd.DataFrame, cfg: HvacModelConfig) -> tuple[str, ...]:
    counts = rows.loc[~rows.censored].groupby("mode").size()
    return tuple(
        sorted(
            str(mode)
            for mode, n in counts.items()
            if n >= 10 and mode != cfg.reference_mode
        )
    )


def _rated_slope(capabilities: HvacCapabilities) -> float | None:
    if not capabilities.rated_points:
        return None
    for supply in sorted({p.supply_temp_c for p in capabilities.rated_points}):
        points = [p for p in capabilities.rated_points if p.supply_temp_c == supply]
        if len({p.source_temp_c for p in points}) < 2:
            continue
        x = np.column_stack([np.ones(len(points)), [p.source_temp_c for p in points]])
        intercept, slope = np.linalg.lstsq(
            x, [p.heat_output_kw for p in points], rcond=None
        )[0]
        at_seven = intercept + 7 * slope
        if at_seven <= 0:
            raise HvacError("rated heat-output curve is nonpositive at 7C")
        return float(slope / at_seven)
    return 0.0


def _fit_one(
    rows: pd.DataFrame,
    cfg: HvacModelConfig,
    capabilities: HvacCapabilities,
    *,
    cancel: Event | None,
    deadline: float | None,
) -> tuple[
    CycleParameters, tuple[tuple[float, float | None], ...], bool, int, tuple[str, ...]
]:
    idle = rows.loc[rows.kind.eq("idle")].copy()
    run = rows.loc[rows.kind.eq("run")].copy()
    imodes, rmodes = _supported_modes(idle, cfg), _supported_modes(run, cfg)
    ix, rx = _design(idle, imodes, idle=True), _design(run, rmodes, idle=False)
    ipen = np.array([0.0] + [1.0] * len(imodes) + [0.0, 4.0, 4.0])
    temp = idle.outdoor_c.to_numpy(dtype=float)
    lower = max(float(np.quantile(rows.outdoor_c, 0.98)) + 1, 10.0)
    grid = np.arange(np.ceil(lower * 2) / 2, 32.00001, 0.5)
    if not len(grid):
        raise HvacError("outdoor history lies outside balance-temperature profile")
    best = None
    profile: list[tuple[float, float | None]] = []
    for balance in grid:
        check_budget(cancel, deadline)
        delta = balance - temp
        # A complete idle with zero demand is impossible in this physical model;
        # censored zero-demand idles have survival one and contribute no information.
        if ((delta <= 0) & ~idle.censored.to_numpy(dtype=bool)).any():
            profile.append((float(balance), None))
            continue
        use = delta > 0
        fit = censored_lognormal_aft(
            ix[use],
            idle.minutes.to_numpy(dtype=float)[use],
            idle.censored.to_numpy(dtype=bool)[use],
            offset=-np.log(delta[use]),
            ridge=ipen,
            cancel=cancel,
            deadline=deadline,
        )
        if not fit.converged:
            profile.append((float(balance), None))
            continue
        profile.append((float(balance), fit.log_likelihood))
        if best is None or fit.log_likelihood > best[1].log_likelihood:
            best = (float(balance), fit)
    if best is None:
        raise HvacError("idle_aft_nonconvergence")
    balance, ifit = best
    provisional = CycleParameters(
        balance, imodes, tuple(ifit.coefficients), ifit.sigma, 1.0, 0.0, (), (), 1.0
    )
    a = provisional.rates(
        run.outdoor_c.to_numpy(),
        run["mode"].to_numpy(),
        run.sun.to_numpy(),
        run.hour_sin.to_numpy(),
        run.hour_cos.to_numpy(),
    )[0]
    max_a = float(np.max(a))
    if max_a <= 0:
        raise HvacError("run history has no positive heat demand")
    fixed_slope = _rated_slope(capabilities)
    slopes = [fixed_slope] if fixed_slope is not None else np.arange(0, 0.04001, 0.005)
    run_best = None
    for slope in slopes:
        relative = np.clip(1 + float(slope) * (run.outdoor_c.to_numpy() - 7), 0.4, 1.3)
        for rho in np.geomspace(1.05 * max_a, 20 * max_a, 40):
            check_budget(cancel, deadline)
            r = rho * relative
            rate = np.maximum(
                r - a, OVERLOAD_MARGIN * r
            )  # fit-only; never a simulation floor
            fit = censored_lognormal_aft(
                rx,
                run.minutes.to_numpy(dtype=float),
                run.censored.to_numpy(dtype=bool),
                offset=-np.log(rate),
                ridge=np.ones(len(rmodes)),
                cancel=cancel,
                deadline=deadline,
            )
            if fit.converged and (
                run_best is None or fit.log_likelihood > run_best[0].log_likelihood
            ):
                run_best = (
                    fit,
                    float(rho),
                    float(slope),
                    int(np.sum(a >= (1 - OVERLOAD_MARGIN) * r)),
                )
    if run_best is None:
        raise HvacError("run_aft_nonconvergence")
    rfit, rho, slope, overloads = run_best
    all_modes = {str(m) for m in rows["mode"].unique() if m != cfg.reference_mode}
    unidentified = tuple(sorted((all_modes - set(imodes)) | (all_modes - set(rmodes))))
    recovery = run.loc[run.first_after_off & ~run.censored, "minutes"]
    ordinary = run.loc[~run.first_after_off & ~run.censored, "minutes"]
    effect = (
        float(np.log(recovery).mean() - np.log(ordinary).mean())
        if len(recovery) and len(ordinary)
        else 0.0
    )
    multiplier = float(np.exp(effect * len(recovery) / (len(recovery) + 10)))
    model = CycleParameters(
        balance,
        imodes,
        tuple(float(v) for v in ifit.coefficients),
        ifit.sigma,
        rho,
        slope,
        rmodes,
        tuple(float(v) for v in rfit.coefficients),
        rfit.sigma,
        multiplier,
    )
    return (
        model,
        tuple(profile),
        balance in (float(grid[0]), float(grid[-1])),
        overloads,
        unidentified,
    )


def fit_cycles(
    sojourns: pd.DataFrame,
    cfg: HvacModelConfig,
    capabilities: HvacCapabilities,
    *,
    cancel: Event | None = None,
    deadline: float | None = None,
    mode_switches: pd.DataFrame | None = None,
) -> tuple[CycleFit | None, str | None]:
    """Fit physical cycle durations and approved full-plus-bootstrap uncertainty.

    Args:
        sojourns: Extracted spells with censoring and covariates.
        cfg: Mode definitions, parameter-set count and seed.
        capabilities: Declared rating points.
        cancel: Caller cooperative cancellation flag.
        deadline: Absolute monotonic deadline checked between profile points.
        mode_switches: Optional observed command outcome table.

    Returns:
        The fitted model or an explicit insufficient-data fallback reason.

    Raises:
        ValueError: If a likelihood fit fails to converge.
        HvacCancelled: If cancelled.
        HvacTimeout: If the deadline expires.
    """
    check_budget(cancel, deadline)
    needed = ["outdoor_c", "sun", "hour_sin", "hour_cos", "minutes"]
    rows = (
        sojourns.loc[sojourns.kind.isin(["run", "idle"])].dropna(subset=needed).copy()
    )
    if "defrost_minutes" in rows:
        active = rows["minutes"] - np.where(
            rows.kind.eq("run"), rows.defrost_minutes, 0.0
        )
        rows["minutes"] = active
        rows = rows.loc[active > 0].copy()
    if rows.empty:
        return None, "insufficient_cycles"
    counts = (
        rows.loc[~rows.censored & rows["mode"].eq(cfg.reference_mode)]
        .groupby("kind")
        .size()
    )
    if counts.get("run", 0) < 40 or counts.get("idle", 0) < 40:
        return None, "insufficient_cycles"
    if float(np.diff(np.quantile(rows.outdoor_c, [0.1, 0.9]))[0]) < 3:
        return None, "narrow_outdoor_range"
    full, profile, edge, overloads, unidentified = _fit_one(
        rows, cfg, capabilities, cancel=cancel, deadline=deadline
    )
    sets = [full]
    local_days = (
        pd.to_datetime(rows.start, utc=True).dt.tz_convert(cfg.timezone).dt.date
    )
    days = sorted(local_days.unique())
    bootstrap_edges = 0
    for b in range(1, cfg.n_parameter_sets):
        check_budget(cancel, deadline)
        rng = np.random.default_rng(cfg.seed + b)
        sampled = rng.choice(len(days), len(days), replace=True)
        boot = pd.concat(
            [rows.loc[local_days.eq(days[int(i)])] for i in sampled], ignore_index=True
        )
        model, _, boot_edge, _, _ = _fit_one(
            boot, cfg, capabilities, cancel=cancel, deadline=deadline
        )
        bootstrap_edges += int(boot_edge)
        sets.append(model)
    run_rows = sojourns.loc[sojourns.kind.eq("run")].copy()
    counts = run_rows.get("defrost_interruptions", pd.Series(0.0, index=run_rows.index))
    overall = float(counts.mean()) if len(counts) else 0.0
    temperatures = run_rows.outdoor_c.to_numpy()
    bins = np.where(temperatures < 3, 0, np.where(temperatures <= 7, 1, 2))
    rates = tuple(
        float((counts.loc[bins == i].sum() + 20 * overall) / ((bins == i).sum() + 20))
        for i in range(3)
    )
    lengths = (
        run_rows.get("defrost_minutes", pd.Series(0.0, index=run_rows.index))
        / counts.where(counts > 0)
    ).dropna()
    length = float(lengths.median()) if len(lengths) else 0.0
    active_run = run_rows.minutes - run_rows.get("defrost_minutes", 0.0)
    mean_active = float(active_run.mean())
    hourly_rates = tuple(
        (
            float(
                60
                * (counts.loc[bins == i].sum() + 20 * overall)
                / (active_run.loc[bins == i].sum() + 20 * mean_active)
            )
            if mean_active > 0
            else 0.0
        )
        for i in range(3)
    )
    effects = []
    if mode_switches is not None and len(mode_switches):
        for keys, group in mode_switches.groupby(["from_mode", "to_mode", "state"]):
            group = group.loc[group.outcome_min > 0]
            if len(group) < 10 or not (~group.outcome_censored).any():
                continue
            fit = censored_lognormal_aft(
                np.ones((len(group), 1)),
                group.outcome_min.to_numpy(),
                group.outcome_censored.to_numpy(dtype=bool),
                cancel=cancel,
                deadline=deadline,
            )
            if not fit.converged:
                raise HvacError("switch_aft_nonconvergence")
            effects.append(
                SwitchEffect(
                    str(keys[0]),
                    str(keys[1]),
                    str(keys[2]),
                    float(fit.coefficients[0]),
                    fit.sigma,
                    len(group),
                )
            )
    return (
        CycleFit(
            tuple(sets),
            profile,
            edge or bool(bootstrap_edges),
            overloads,
            unidentified,
            (rates[0], rates[1], rates[2]),
            length,
            tuple(effects),
            bootstrap_edges,
            (hourly_rates[0], hourly_rates[1], hourly_rates[2]),
        ),
        None,
    )
