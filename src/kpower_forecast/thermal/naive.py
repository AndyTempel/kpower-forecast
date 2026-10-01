"""Model-free indoor-temperature fallback for scopes without a usable fit."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .model import (
    THERMAL_CONTRACT_VERSION,
    ThermalPrediction,
    ThermalPredictionInterval,
)

NAIVE_SOURCE = "naive_persistence_trend"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("thermal timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def recent_trend_c_per_hour(
    readings: list[tuple[datetime, float]],
    *,
    now: datetime,
    window_hours: float = 3.0,
    min_span_hours: float = 1.0,
    max_abs_c_per_hour: float = 0.5,
) -> float:
    """Least-squares slope of real readings over the recent window.

    Returns 0 when the window has fewer than two readings or too short a span
    to separate a trend from sensor quantisation.
    """
    cutoff = _utc(now) - timedelta(hours=window_hours)
    recent = [
        ((_utc(at) - cutoff).total_seconds() / 3600, value)
        for at, value in readings
        if cutoff <= _utc(at) <= _utc(now) and math.isfinite(value)
    ]
    if len(recent) < 2:
        return 0.0
    times = np.array([item[0] for item in recent])
    values = np.array([item[1] for item in recent])
    if float(np.ptp(times)) < min_span_hours:
        return 0.0
    slope = float(np.polyfit(times, values, 1)[0])
    return float(np.clip(slope, -max_abs_c_per_hour, max_abs_c_per_hour))


def predict_naive(
    *,
    origin: datetime,
    initial_temperature_c: float,
    trend_c_per_hour: float,
    periods: int,
    interval_minutes: int = 15,
    damping_hours: float = 2.0,
    max_change_c: float = 2.0,
    base_width_c: float = 0.3,
    max_width_c: float = 3.0,
) -> ThermalPrediction:
    """Hold the current reading, relaxing toward a damped recent trend.

    ``T(t) = T0 + trend * damping * (1 - exp(-t / damping))``, so the trend
    contributes at most ``trend * damping`` before the forecast flattens. The
    band grows with the square root of the horizon.

    Args:
        origin: Aligned start of the first future interval.
        initial_temperature_c: Fresh real indoor reading.
        trend_c_per_hour: Recent observed slope, already bounded.
        periods: Number of future intervals.
        interval_minutes: Grid step.
        damping_hours: Time constant over which the trend fades.
        max_change_c: Bound on the total trend contribution.
        base_width_c: Band half-width after one hour.
        max_width_c: Band half-width cap.

    Returns:
        A trajectory labelled with the naive outdoor and drive sources.
    """
    if interval_minutes <= 0 or damping_hours <= 0:
        raise ValueError("naive forecast interval and damping must be positive")
    origin_utc = _utc(origin)
    step_seconds = interval_minutes * 60
    if origin_utc.timestamp() % step_seconds != 0:
        raise ValueError("thermal prediction origin is off the UTC grid")
    if periods <= 0:
        raise ValueError("thermal prediction period count is invalid")
    if not math.isfinite(initial_temperature_c) or not math.isfinite(trend_c_per_hour):
        raise ValueError("naive forecast inputs must be finite")
    intervals: list[ThermalPredictionInterval] = []
    for index in range(1, periods + 1):
        hours = index * interval_minutes / 60
        change = (
            trend_c_per_hour * damping_hours * (1 - math.exp(-hours / damping_hours))
        )
        state = initial_temperature_c + max(-max_change_c, min(max_change_c, change))
        width = min(max_width_c, base_width_c * math.sqrt(hours))
        intervals.append(
            ThermalPredictionInterval(
                timestamp=origin_utc + timedelta(seconds=index * step_seconds),
                indoor_temperature_c=state,
                lower_temperature_c=state - width,
                upper_temperature_c=state + width,
            )
        )
    return ThermalPrediction(
        origin=origin_utc,
        model_contract_version=THERMAL_CONTRACT_VERSION,
        outdoor_source=NAIVE_SOURCE,
        hvac_drive_source=NAIVE_SOURCE,
        intervals=intervals,
    )
