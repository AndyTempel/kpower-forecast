from datetime import datetime, timezone

import pandas as pd
import pytest

from kpower_forecast.ml.baselines import (
    evaluate_local_slot_baseline,
    local_slot_weekday_class_median,
)


def test_local_slot_baseline_uses_only_pre_origin_history() -> None:
    history = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-08-03T10:00:00Z",
                    "2026-08-04T10:00:00Z",
                    "2026-08-05T10:00:00Z",
                    "2026-08-10T10:00:00Z",
                ]
            ),
            "y": [1.0, 3.0, 100.0, 9999.0],
        }
    )

    result = local_slot_weekday_class_median(
        history,
        origin=datetime(2026, 8, 10, 10, tzinfo=timezone.utc),
        periods=1,
        interval_minutes=15,
    )

    assert result["ds"].tolist() == [pd.Timestamp("2026-08-10T10:00:00Z")]
    assert result["yhat"].tolist() == [3.0]


def test_local_slot_baseline_respects_local_weekday_class() -> None:
    history = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-08-07T22:00:00Z",  # Saturday 00:00 in Ljubljana
                    "2026-08-08T22:00:00Z",  # Sunday 00:00 in Ljubljana
                    "2026-08-09T22:00:00Z",  # Monday 00:00 in Ljubljana
                ]
            ),
            "y": [8.0, 10.0, 2.0],
        }
    )

    result = local_slot_weekday_class_median(
        history,
        origin=datetime(2026, 8, 15, 22, tzinfo=timezone.utc),
        periods=1,
        interval_minutes=15,
        timezone="Europe/Ljubljana",
    )

    assert result["yhat"].tolist() == [9.0]


def test_rolling_origin_baseline_reports_exact_metrics() -> None:
    ds = pd.date_range("2026-07-01T00:00:00Z", periods=24 * 4 * 10, freq="15min")
    history = pd.DataFrame({"ds": ds, "y": [2.0] * len(ds)})

    metrics = evaluate_local_slot_baseline(
        history,
        origins=[datetime(2026, 7, 9, tzinfo=timezone.utc)],
        horizon_periods=[4],
        interval_minutes=15,
    )[4]

    assert metrics.samples == 4
    assert metrics.mae == 0.0
    assert metrics.wape_pct == 0.0
    assert metrics.bias_pct == 0.0
    assert metrics.p90_absolute_error == pytest.approx(0.0)


def test_rolling_origin_baseline_reuses_generator_for_each_horizon() -> None:
    ds = pd.date_range("2026-07-01T00:00:00Z", periods=24 * 4 * 10, freq="15min")
    history = pd.DataFrame({"ds": ds, "y": [2.0] * len(ds)})
    origin_values = (
        origin.to_pydatetime()
        for origin in pd.date_range("2026-07-08T00:00:00Z", periods=2, freq="6h")
    )

    metrics = evaluate_local_slot_baseline(
        history,
        origins=origin_values,
        horizon_periods=[4, 8],
        interval_minutes=15,
    )

    assert metrics[4].samples == 8
    assert metrics[8].samples == 16
