"""Leakage-safe baseline forecasts and rolling-origin evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from kpower_forecast.ml.alignment import ForecastAlignmentError, prediction_origin

BASELINE_NAME = "local_slot_weekday_class_median"


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    """Aggregate errors for one rolling-origin horizon."""

    samples: int
    mae: float
    wape_pct: float
    bias_pct: float
    p50_absolute_error: float
    p90_absolute_error: float

    def as_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable metric values."""
        return asdict(self)


def local_slot_weekday_class_median(
    history: pd.DataFrame,
    *,
    origin: datetime,
    periods: int,
    interval_minutes: int,
    timezone: str = "UTC",
    lookback_days: int = 56,
) -> pd.DataFrame:
    """Forecast from historical medians for local-time slot and weekday class.

    Only observations strictly before ``origin`` are considered, making the
    baseline safe for fallback inference and rolling-origin evaluation.
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    start = prediction_origin(origin, interval_minutes)
    prepared = _prepare_history(history)
    cutoff = start - pd.Timedelta(days=lookback_days)
    prepared = prepared[(prepared["ds"] < start) & (prepared["ds"] >= cutoff)].copy()
    if prepared.empty:
        raise ForecastAlignmentError("baseline has no observations before origin")

    prepared_local = prepared["ds"].dt.tz_convert(timezone)
    prepared["slot"] = (prepared_local.dt.hour * 60 + prepared_local.dt.minute).astype(
        int
    )
    prepared["weekend"] = prepared_local.dt.dayofweek.ge(5)
    class_medians = prepared.groupby(["slot", "weekend"])["y"].median()
    slot_medians = prepared.groupby("slot")["y"].median()
    global_median = float(prepared["y"].median())

    future = pd.date_range(
        start=start,
        periods=periods,
        freq=f"{interval_minutes}min",
        tz="UTC",
    )
    future_local = future.tz_convert(timezone)
    values: list[float] = []
    for timestamp in future_local:
        slot = timestamp.hour * 60 + timestamp.minute
        weekend = timestamp.dayofweek >= 5
        value = class_medians.get(
            (slot, weekend), slot_medians.get(slot, global_median)
        )
        values.append(max(float(value), 0.0))
    return pd.DataFrame({"ds": future, "yhat": values})


def evaluate_local_slot_baseline(
    history: pd.DataFrame,
    *,
    origins: Iterable[datetime],
    horizon_periods: Iterable[int],
    interval_minutes: int,
    timezone: str = "UTC",
) -> dict[int, BaselineMetrics]:
    """Evaluate the baseline over exact rolling-origin horizons."""
    prepared = _prepare_history(history).set_index("ds")
    origin_values = list(origins)
    results: dict[int, BaselineMetrics] = {}
    for periods in horizon_periods:
        actual_values: list[float] = []
        predicted_values: list[float] = []
        for origin_value in origin_values:
            origin = prediction_origin(origin_value, interval_minutes)
            expected = pd.date_range(
                start=origin,
                periods=periods,
                freq=f"{interval_minutes}min",
                tz="UTC",
            )
            actual = prepared.reindex(expected)["y"]
            if actual.isna().any():
                continue
            prediction = local_slot_weekday_class_median(
                prepared.reset_index(),
                origin=origin.to_pydatetime(),
                periods=periods,
                interval_minutes=interval_minutes,
                timezone=timezone,
            )
            actual_values.extend(actual.astype(float).tolist())
            predicted_values.extend(prediction["yhat"].astype(float).tolist())
        if not actual_values:
            raise ForecastAlignmentError(
                f"no complete evaluation windows for {periods} periods"
            )
        results[periods] = _metrics(actual_values, predicted_values)
    return results


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    if not {"ds", "y"}.issubset(history.columns):
        raise ValueError("history must contain 'ds' and 'y' columns")
    prepared = history[["ds", "y"]].copy()
    prepared["ds"] = pd.to_datetime(prepared["ds"], utc=True)
    prepared["y"] = pd.to_numeric(prepared["y"], errors="coerce")
    prepared = prepared.dropna().sort_values("ds")
    return prepared.drop_duplicates(subset="ds", keep="last").reset_index(drop=True)


def _metrics(actual: list[float], predicted: list[float]) -> BaselineMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    errors = predicted_values - actual_values
    absolute_errors = np.abs(errors)
    denominator = float(np.abs(actual_values).sum())
    if denominator <= 0:
        raise ValueError("WAPE and bias require non-zero actual energy")
    return BaselineMetrics(
        samples=len(actual_values),
        mae=float(absolute_errors.mean()),
        wape_pct=float(absolute_errors.sum() / denominator * 100.0),
        bias_pct=float(errors.sum() / denominator * 100.0),
        p50_absolute_error=float(np.quantile(absolute_errors, 0.5)),
        p90_absolute_error=float(np.quantile(absolute_errors, 0.9)),
    )
