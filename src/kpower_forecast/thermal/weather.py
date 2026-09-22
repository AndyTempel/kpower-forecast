"""Strict UTC outdoor-temperature alignment for passive thermal modelling.

Weather interpolation applies only to the exogenous outdoor input. Indoor
observations remain exactly the sparse source observations supplied by EMS.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

import numpy as np
import pandas as pd

from kpower_forecast.weather_client import WeatherClient


def _weather_series(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite, unique UTC weather samples without filling any gaps."""
    if not {"ds", "temperature_2m"}.issubset(frame.columns) or frame.empty:
        raise ValueError("weather lacks UTC timestamps or temperature_2m")
    raw = frame["ds"]
    if any(pd.Timestamp(value).tzinfo is None for value in raw):
        raise ValueError("weather timestamps must be timezone-aware")
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw, utc=True))
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("weather timestamps must be unique and ordered")
    values = pd.to_numeric(frame["temperature_2m"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("weather temperature contains missing or non-finite values")
    # Pandas 3 may back arrays with microseconds. Timestamp.value is always ns.
    return np.array([item.value / 1e9 for item in timestamps]), values


def mean_outdoor_temperature(
    frame: pd.DataFrame,
    *,
    start_at: datetime,
    end_at: datetime,
    max_sample_gap: timedelta = timedelta(hours=1),
) -> float:
    """Integrate linearly between real weather samples over an arbitrary span.

    Args:
        frame: Ordered UTC weather samples with ``ds`` and ``temperature_2m``.
        start_at: Real indoor transition start.
        end_at: Real indoor transition end.
        max_sample_gap: Largest acceptable weather sampling interval.

    Returns:
        Elapsed-time-weighted outdoor temperature in Celsius.

    Raises:
        ValueError: For missing coverage, large weather gaps, or invalid time.
    """
    if start_at.tzinfo is None or end_at.tzinfo is None:
        raise ValueError("thermal transition timestamps must be timezone-aware")
    start = start_at.astimezone(timezone.utc).timestamp()
    end = end_at.astimezone(timezone.utc).timestamp()
    if end <= start:
        raise ValueError("thermal transition must have positive elapsed time")
    times, values = _weather_series(frame)
    if start < times[0] or end > times[-1]:
        raise ValueError("historical weather does not cover thermal transition")
    left = max(int(np.searchsorted(times, start)) - 1, 0)
    right = min(int(np.searchsorted(times, end, side="right")), len(times) - 1)
    if np.any(np.diff(times[left : right + 1]) > max_sample_gap.total_seconds()):
        raise ValueError("historical weather has a material gap")
    interior = times[(times > start) & (times < end)]
    knots = np.concatenate(([start], interior, [end]))
    interpolated = np.interp(knots, times, values)
    return float(np.trapezoid(interpolated, knots) / (end - start))


def exact_future_outdoor_grid(
    frame: pd.DataFrame,
    *,
    origin: datetime,
    periods: int,
    interval_minutes: int,
) -> list[float]:
    """Select a contiguous future weather grid without extrapolation or fill."""
    if origin.tzinfo is None or origin.utcoffset() is None:
        raise ValueError("thermal prediction origin must be timezone-aware")
    if periods < 1 or interval_minutes < 1:
        raise ValueError("future weather grid dimensions must be positive")
    times, values = _weather_series(frame)
    expected = np.array(
        [
            item.value / 1e9
            for item in pd.date_range(
                start=origin.astimezone(timezone.utc),
                periods=periods,
                freq=f"{interval_minutes}min",
                tz="UTC",
            )
        ]
    )
    positions = np.searchsorted(times, expected)
    if np.any(positions >= len(times)) or not np.array_equal(
        times[positions], expected
    ):
        raise ValueError("future weather is missing an exact UTC forecast interval")
    return cast(list[float], values[positions].tolist())


def fetch_historical_weather(
    client: WeatherClient, *, start_at: datetime, end_at: datetime
) -> pd.DataFrame:
    """Fetch archive weather with adjacent dates for endpoint integration."""
    start = start_at.astimezone(timezone.utc).date() - timedelta(days=1)
    end = end_at.astimezone(timezone.utc).date() + timedelta(days=1)
    return client.fetch_historical(start, end, strict=True)


def fetch_future_weather(
    client: WeatherClient, *, periods: int, interval_minutes: int
) -> pd.DataFrame:
    """Fetch enough package-owned forecast weather for the requested horizon."""
    days = max(1, (periods * interval_minutes + 1439) // 1440 + 1)
    return client.fetch_forecast(days=days, strict=True)
