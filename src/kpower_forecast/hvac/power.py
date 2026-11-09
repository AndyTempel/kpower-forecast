"""Meter-scope electrical power, observed standby and interruption dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._stats import huber_regression
from .cycles import modes_at
from .types import HvacError, HvacModelConfig, MeteredScope


@dataclass(frozen=True)
class PowerFit:
    """One run-level power regression and measured low-power states."""

    coefficients: tuple[float, ...]
    uses_supply: bool
    residual_sd_w: float
    start_ramp: tuple[float, float, float]
    idle_by_mode: Mapping[str, float]
    idle_default_w: float
    off_w: float
    defrost_w: float | None
    metered_scope: MeteredScope
    diagnostics: tuple[str, ...]
    supply_reference_c: float = 35.0
    defrost_running_share: float = 0.0

    def run_power(
        self, outdoor_c: ArrayLike, supply_c: ArrayLike = 35.0
    ) -> NDArray[np.float64]:
        """Evaluate mean measured run power, never a thermal heat-output estimate.

        Args:
            outdoor_c: Outdoor temperatures.
            supply_c: Supply/setpoint temperature when fitted, otherwise unused.

        Returns:
            Nonnegative run electrical watts, in the declared meter scope.
        """
        t, s = np.broadcast_arrays(
            np.asarray(outdoor_c, dtype=float), np.asarray(supply_c, dtype=float)
        )
        value = self.coefficients[0] + self.coefficients[1] * (t - 7)
        if self.uses_supply:
            value = value + self.coefficients[2] * (s - 35)
        return np.asarray(np.maximum(value, 0.0))


def fit_power(
    frame: pd.DataFrame,
    sojourns: pd.DataFrame,
    mode_events: pd.DataFrame,
    cfg: HvacModelConfig,
    scope: MeteredScope,
) -> PowerFit:
    """Fit run power and observed ramps/standby/defrost without zero-filling.

    Args:
        frame: Qualified minute observations and running state.
        sojourns: Extracted runs, idles and OFF spells.
        mode_events: Causal command history.
        cfg: Caller mode configuration.
        scope: Components measured by the meter.

    Returns:
        Regression and state powers, with fallback evidence for unobserved OFF/ramp.

    Raises:
        ValueError: If required run or idle power cannot be estimated.
    """
    runs = (
        sojourns.loc[sojourns.kind.eq("run")]
        .dropna(subset=["power_w", "outdoor_c"])
        .copy()
    )
    if len(runs) < 3:
        raise HvacError("insufficient_run_power")
    use_supply = bool("supply_c" in runs and runs.supply_c.notna().sum() >= 3)
    if use_supply:
        runs = runs.dropna(subset=["supply_c"])
    columns = [np.ones(len(runs)), runs.outdoor_c.to_numpy() - 7]
    if use_supply:
        columns.append(runs.supply_c.to_numpy() - 35)
    design = np.column_stack(columns)
    coefficients = huber_regression(design, runs.power_w.to_numpy())
    residual = runs.power_w.to_numpy() - design @ coefficients
    state = modes_at(pd.DatetimeIndex(frame.index), mode_events, cfg.reference_mode)
    idles = frame.loc[
        frame.running.eq(0) & ~state.isin(cfg.off_modes) & frame.power_w.notna()
    ]
    if idles.empty:
        raise HvacError("no_measured_idle_power")
    default = float(idles.power_w.median())
    by_mode = {
        str(m): float(idles.loc[state.loc[idles.index].eq(m), "power_w"].median())
        for m in state.loc[idles.index].unique()
    }
    off = frame.loc[state.isin(cfg.off_modes), "power_w"].dropna()
    diagnostics = []
    if off.empty:
        diagnostics.append("off_power_from_idle")
    ramps: list[list[float]] = [[], [], []]
    defrost = []
    defrost_on: list[float] = []
    for row in runs.itertuples():
        start = cast(pd.Timestamp, row.start)
        end = cast(pd.Timestamp, row.end)
        level = float(cast(float, row.power_w))
        if not bool(row.censored) and level > 0:
            for i in range(3):
                at = start + pd.Timedelta(minutes=i)
                if at in frame.index and pd.notna(frame.loc[at, "power_w"]):
                    ramps[i].append(
                        float(cast(float, frame.loc[at, "power_w"])) / level
                    )
        inside = frame.loc[(frame.index >= start) & (frame.index < end)]
        mask = inside.running.eq(0)
        if "defrost" in inside:
            mask |= inside.defrost.eq(1)
        defrost.extend(inside.loc[mask, "power_w"].dropna().tolist())
        defrost_on.extend(inside.loc[mask, "running"].dropna().tolist())
    for i, values in enumerate(ramps):
        if not values:
            diagnostics.append(f"start_ramp_unidentified:{i}")
    ramp = tuple(float(np.mean(values)) if values else 1.0 for values in ramps)
    return PowerFit(
        tuple(float(x) for x in coefficients),
        use_supply,
        float(np.std(residual, ddof=1)),
        (ramp[0], ramp[1], ramp[2]),
        by_mode,
        default,
        float(off.median()) if len(off) else default,
        float(np.median(defrost)) if defrost else None,
        scope,
        tuple(diagnostics),
        float(runs.supply_c.mean()) if use_supply else 35.0,
        float(np.mean(defrost_on)) if defrost_on else 0.0,
    )
