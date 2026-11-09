"""Equipment-family contract and explicit energy-only reserved-family fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._stats import check_budget
from .cycles import modes_at
from .types import (
    FitResult,
    HvacError,
    HvacFamily,
    HvacForecastRequest,
    HvacModelConfig,
    HvacTrainingData,
    MinutePaths,
)


class EquipmentModel(Protocol):
    """Family plug-in interface; fitting returns a candidate without installation."""

    family: HvacFamily

    def fit(
        self,
        data: HvacTrainingData,
        cfg: HvacModelConfig,
        *,
        as_of: datetime,
        cancel: Event | None = None,
    ) -> FitResult:
        """Return a candidate from inputs available at as_of."""
        ...

    def simulate(
        self,
        request: HvacForecastRequest,
        fit: FitResult,
        cfg: HvacModelConfig,
        *,
        cancel: Event | None = None,
    ) -> MinutePaths:
        """Return the full in-memory ensemble, without representative reduction."""
        ...


class FixedSpeedModel:
    """Alternating physical log-normal idle/run process with explicit overload."""

    family: HvacFamily = HvacFamily.FIXED_SPEED

    def fit(
        self,
        data: HvacTrainingData,
        cfg: HvacModelConfig,
        *,
        as_of: datetime,
        cancel: Event | None = None,
    ) -> FitResult:
        """Fit a candidate without changing active state or disk.

        Args:
            data: Evidence snapshot.
            cfg: Caller configuration.
            as_of: Causal fit origin.
            cancel: Cooperative cancellation.

        Returns:
            Candidate coefficients and quality diagnostics.
        """
        from .model import _fit_candidate

        return _fit_candidate(data, cfg, as_of=as_of, cancel=cancel)

    def simulate(
        self,
        request: HvacForecastRequest,
        fit: FitResult,
        cfg: HvacModelConfig,
        *,
        cancel: Event | None = None,
    ) -> MinutePaths:
        """Simulate a candidate without installing it.

        Args:
            request: Causal origin, weather and known schedule.
            fit: Candidate coefficients.
            cfg: Compatible caller configuration.
            cancel: Cooperative cancellation.

        Returns:
            Full ensemble; state is unknown for energy-only fallbacks.
        """
        from .model import _predict_candidate

        return _predict_candidate(request, fit, cfg, cancel=cancel).minute_paths


class EnergyOnlyModel(FixedSpeedModel):
    """T0 hourly ridge energy with unknown compressor state."""

    family: HvacFamily = HvacFamily.UNKNOWN

    def fit(
        self,
        data: HvacTrainingData,
        cfg: HvacModelConfig,
        *,
        as_of: datetime,
        cancel: Event | None = None,
    ) -> FitResult:
        """Fit T0 explicitly; its absence of compressor probabilities is preserved.

        Args:
            data: Evidence snapshot.
            cfg: Caller configuration.
            as_of: Causal fit origin.
            cancel: Cooperative cancellation.

        Returns:
            Energy-only candidate.
        """
        from .model import _fit_candidate

        return _fit_candidate(data, cfg, as_of=as_of, cancel=cancel, force_energy=True)


class InverterModel(EnergyOnlyModel):
    """Reserved inverter extension; v1 serves T0, flagged family_not_implemented.

    Future continuous power is P=clip(Q(T)/COP(T), P_min, P_max). Below Cap_min,
    cycle at P_min with the same demand shape and hysteresis sojourns. High run
    probability can coexist with low power; energy and run probability are separate.
    """

    family: HvacFamily = HvacFamily.INVERTER


class TwoStageModel(EnergyOnlyModel):
    """Reserved two-stage extension; v1 serves explicitly flagged T0.

    RUN has levels L1 and L2 with learned electrical powers. Stage-up/down are
    demand-conditioned sojourns. A level change is not a compressor start.
    """

    family: HvacFamily = HvacFamily.TWO_STAGE


class MultiStageModel(EnergyOnlyModel):
    """Reserved multi-stage extension; v1 serves explicitly flagged T0.

    RUN has levels L1...Ln with learned electrical powers. Stage-up/down are
    demand-conditioned sojourns. A level change is not a compressor start.
    """

    family: HvacFamily = HvacFamily.MULTI_STAGE


@dataclass(frozen=True)
class EnergyFit:
    """Hourly energy shape and empirical minute residual calibration."""

    t_bal_c: float
    modes: tuple[str, ...]
    coefficients: tuple[float, ...]
    residual_profiles_w: tuple[tuple[float, ...], ...]
    calibration_counts: tuple[int, ...]

    def mean_power(
        self,
        temperature: NDArray[np.float64],
        sun: NDArray[np.float64],
        modes: NDArray[np.str_],
    ) -> NDArray[np.float64]:
        """Evaluate nonnegative electrical watts with no calendar intercepts.

        Args:
            temperature: Selected site outdoor C.
            sun: Shortwave radiation kW/m².
            modes: Operating category per minute.

        Returns:
            Nonnegative expected curve before empirical path noise.
        """
        b = self.coefficients
        value = b[0] + b[1] * np.maximum(self.t_bal_c - temperature, 0) + b[2] * sun
        for i, label in enumerate(self.modes):
            value = value + b[3 + i] * (modes == label)
        return np.asarray(np.maximum(value * 1000, 0), dtype=float)


def fit_energy(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    cfg: HvacModelConfig,
    *,
    t_bal_c: float | None = None,
    cancel: Event | None = None,
    deadline: float | None = None,
) -> EnergyFit:
    """Fit hourly energy on full observed hours; target gaps remain excluded.

    Args:
        frame: Qualified minute target, outdoor and sun. Bridged targets are excluded.
        events: Origin-sliced mode history.
        cfg: Reference mode and timezone.
        t_bal_c: Existing cycle balance temperature, otherwise profiled.
        cancel: Cooperative cancellation.
        deadline: Absolute monotonic deadline.

    Returns:
        Ridge coefficients and complete-hour empirical residual profiles, which
        retain dependence for every supported output duration (1/5/15/30/60 min).

    Raises:
        ValueError: When too few full observed hours are available.
    """
    rows = frame[["power_w", "outdoor_c", "sun"]].copy()
    rows["mode"] = modes_at(pd.DatetimeIndex(rows.index), events, cfg.reference_mode)
    valid = rows[["power_w", "outdoor_c", "sun"]].notna().all(axis=1)
    complete = valid.resample("h").sum().eq(60)
    rows = rows.loc[
        pd.DatetimeIndex(rows.index).floor("h").isin(complete.index[complete])
    ]
    if len(rows) < 3 * 60:
        raise HvacError("insufficient_energy_history")
    hourly = rows[["power_w", "outdoor_c", "sun"]].resample("h").mean().dropna()
    supported = tuple(
        sorted(
            str(m)
            for m, count in rows["mode"].value_counts().items()
            if m != cfg.reference_mode and count >= 60
        )
    )
    fractions = [
        rows["mode"].eq(m).resample("h").mean().reindex(hourly.index).to_numpy()
        for m in supported
    ]
    temperatures = hourly.outdoor_c.to_numpy(dtype=float)
    candidates = (
        [t_bal_c]
        if t_bal_c is not None
        else np.arange(
            np.ceil(max(float(np.quantile(temperatures, 0.98)) + 1, 10) * 2) / 2,
            32.01,
            0.5,
        ).tolist()
    )
    if not candidates:
        raise HvacError("energy_balance_grid_empty")
    best: tuple[float, float, NDArray[np.float64]] | None = None
    y = hourly.power_w.to_numpy(dtype=float) / 1000
    for balance in candidates:
        check_budget(cancel, deadline)
        x = np.column_stack(
            [
                np.ones(len(hourly)),
                np.maximum(balance - temperatures, 0),
                hourly.sun.to_numpy(),
                *fractions,
            ]
        )
        penalty = np.diag([0.0, 1.0, 1.0, *([1.0] * len(supported))])
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        loss = float(np.sum((y - x @ beta) ** 2) + beta @ penalty @ beta)
        if best is None or loss < best[0]:
            best = loss, float(balance), beta
    assert best is not None
    provisional = EnergyFit(
        best[1], supported, tuple(float(v) for v in best[2]), (), ()
    )
    residual = rows.power_w.to_numpy() - provisional.mean_power(
        rows.outdoor_c.to_numpy(), rows.sun.to_numpy(), rows["mode"].to_numpy(dtype=str)
    )
    profiles = residual.reshape(-1, 60)
    counts = tuple(int(profiles.size // length) for length in (1, 5, 15, 30, 60))
    return EnergyFit(
        best[1],
        supported,
        provisional.coefficients,
        tuple(tuple(float(v) for v in row) for row in profiles),
        counts,
    )


def energy_paths(
    fit: EnergyFit,
    mean_w: NDArray[np.float64],
    rng: np.random.Generator,
    start: datetime,
) -> MinutePaths:
    """Sample observed hourly residual profiles and preserve unknown compressor state.

    Args:
        fit: Energy coefficients and empirical residual profiles.
        mean_w: Expected per-path minute power, including known schedule categories.
        rng: Prediction-owned generator.
        start: UTC minute-start anchor.

    Returns:
        Nonnegative full ensemble with running and starts both None. Display
        representatives are intentionally omitted by the public forecast wrapper.
    """
    n, horizon = mean_w.shape
    profiles = np.asarray(fit.residual_profiles_w)
    selected = rng.integers(0, len(profiles), size=(n, int(np.ceil(horizon / 60))))
    residual = profiles[selected].reshape(n, -1)[:, :horizon]
    return MinutePaths(
        start, np.maximum(mean_w + residual, 0).astype(np.float32), None, None
    )
