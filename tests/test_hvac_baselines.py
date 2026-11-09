"""Registered baselines retain causal gaps, source priority and evidence limits."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac import (
    BundleSnapshot,
    HistoryPolicy,
    HvacCapabilities,
    HvacModelConfig,
    HvacTrainingData,
)
from kpower_forecast.hvac._baselines import (
    legacy_baseline,
    r1c_probabilities,
    simple_baselines,
)
from kpower_forecast.hvac.evaluation import regular_grid

T0 = datetime(2026, 11, 9, 12, tzinfo=UTC)


@pytest.fixture
def view():
    owner = json.loads(
        (Path(__file__).parent / "fixtures/hvac_owner_policy.json").read_text()
    )
    policy = HistoryPolicy(
        version=owner["version"],
        scope="direct_device",
        settings=owner["settings"],
        measured_valid_qualities=owner["measured_valid_qualities"],
    )
    index = pd.date_range(T0 - timedelta(days=2), periods=2880, freq="min")
    power = 1000 + (np.arange(len(index)) // 15 % 96) * 10
    minutes = pd.DataFrame(
        {"power_w": power, "power_w__measured_valid": True, "power_w__quality": "good"},
        index=index,
    )
    data = HvacTrainingData(
        HvacCapabilities(family="fixed_speed", fingerprint="test"),
        minutes,
        pd.DataFrame(),
        pd.DataFrame(),
        index[0].to_pydatetime(),
        {},
        policy,
    )
    return BundleSnapshot(
        T0,
        data,
        pd.DataFrame(),
        (),
        None,
        None,
        "NORMAL",
        None,
        pd.DataFrame(),
        {},
        {},
        {},
    )


def test_naive_forecasts_repeat_only_available_past_data_and_preserve_unknown_quarters(
    view,
):
    cfg = HvacModelConfig(model_id="test", timezone="UTC", ingest_lag_min=2)
    grid = regular_grid(T0, minutes=15, hours=72)
    forecast = simple_baselines(view, cfg, grid)
    b1, b2 = forecast["B1"], forecast["B2"]
    assert np.unique(b1.expected_kwh).size == 1
    np.testing.assert_array_equal(b2.expected_kwh[:96], b2.expected_kwh[96:192])
    assert np.isnan(b2.expected_kwh[95])  # yesterday's ingestion tail is not known
    assert b2.expected_kwh[0] == 0.25 and b2.expected_kwh[1] == 0.2525
    future = view.training.minutes.iloc[-60:].copy()
    future.index += timedelta(days=1)
    future["power_w"] = 1e8
    changed = replace(
        view,
        training=replace(
            view.training, minutes=pd.concat([view.training.minutes, future])
        ),
    )
    again = simple_baselines(changed, cfg, grid)
    np.testing.assert_array_equal(b1.expected_kwh, again["B1"].expected_kwh)
    np.testing.assert_array_equal(b2.expected_kwh, again["B2"].expected_kwh)


def test_legacy_priority_and_development_proxy_never_substitute_for_exact_live_flags(
    view,
):
    starts = pd.date_range(T0, periods=480, freq="15min")
    daily = pd.DataFrame(
        {
            "origin": T0 - timedelta(hours=1),
            "horizon": "daily",
            "interval_start": starts,
            "expected_w": 2000.0,
            "lower_w": 500.0,
            "upper_w": 5000.0,
            "band_raised": False,
        }
    )
    daily.loc[4, "upper_w"] = 4000.0
    daily.loc[4, "band_raised"] = True
    hourly = pd.DataFrame(
        {
            "origin": T0 - timedelta(minutes=1),
            "horizon": "hourly",
            "interval_start": starts[:4],
            "expected_w": 1000.0,
            "lower_w": 100.0,
            "upper_w": [4000.0, 3000.0, 3000.0, 3000.0],
            "band_raised": False,
        }
    )
    metadata = pd.DataFrame(
        {
            "origin": [T0 - timedelta(hours=1), T0 - timedelta(minutes=1)],
            "horizon": ["daily", "hourly"],
            "applied": True,
            "raised_upper_intervals": 1,
            "run_power_w": 4000.0,
        }
    )
    all_runs = pd.concat([hourly, daily], ignore_index=True)
    view = replace(
        view,
        baselines={
            "served_forecast.parquet": all_runs,
            "legacy_shadow.parquet": all_runs,
            "legacy_awareness.parquet": metadata,
        },
    )
    grid = regular_grid(T0, minutes=15, hours=72)
    old = legacy_baseline(view, grid)
    assert old.expected_kwh[0] == 0.25
    assert old.expected_kwh[4] == 0.5  # legacy hourly covers1h, then daily takes over
    assert old.planner_w[0] == 4000.0 and old.planner_w[1] == 1000.0
    exact = legacy_baseline(view, grid, live=True)
    assert exact.planner_w[0] == 1000.0  # exactFalse wins over the count-based proxy
    assert exact.planner_w[4] == 4000.0
    missing = all_runs.drop(columns="band_raised")
    absent = legacy_baseline(
        replace(view, baselines={**view.baselines, "legacy_shadow.parquet": missing}),
        grid,
        live=True,
    )
    assert np.isnan(absent.planner_w).all()
    partial = all_runs.drop(
        index=2
    )  # lost hourly point must not borrow daily in that hour
    damaged = legacy_baseline(
        replace(view, baselines={**view.baselines, "served_forecast.parquet": partial}),
        grid,
    )
    assert np.isnan(damaged.expected_kwh[2])
    assert np.isnan(
        damaged.planner_w[:4]
    ).all()  # incomplete publication cannot prove flags
    assert damaged.expected_kwh[4] == 0.5


def test_r1c_comparison_requires_identical_origin_and_keeps_null_probabilities(view):
    rows = pd.DataFrame(
        {
            "origin": [T0, T0, T0 - timedelta(minutes=15)],
            "step_start": [T0, T0 + timedelta(minutes=5), T0 + timedelta(minutes=10)],
            "p_run": [0.6, None, 0.9],
        }
    )
    grid = regular_grid(T0, minutes=5, hours=1)
    result = r1c_probabilities(
        replace(view, baselines={"r1c_replay.parquet": rows}), grid
    )
    assert result[0] == 0.6 and np.isnan(result[1:]).all()
