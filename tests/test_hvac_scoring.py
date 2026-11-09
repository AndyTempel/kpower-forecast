"""Scoring preserves ensemble mass, actual gaps and unknown compressor state."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac._baselines import BaselineForecast
from kpower_forecast.hvac._block_scores import block_scores
from kpower_forecast.hvac._scoring import (
    cadence_scores,
    energy_scores,
    quarter_scores,
    run_scores,
)
from kpower_forecast.hvac.aggregate import interval_stats
from kpower_forecast.hvac.evaluation import regular_grid
from kpower_forecast.hvac.settlement import SettlementWindow
from kpower_forecast.hvac.types import HvacFamily, HvacForecast, InputTier, MinutePaths

ORIGIN = datetime(2026, 11, 9, tzinfo=UTC)


@pytest.fixture
def evidence():
    # Display paths deliberately contain only zeros; scoring must use all 256.
    power = np.repeat(np.array([0.0, 2000.0], dtype=np.float32), 128)
    power = np.repeat(power[:, None], 24 * 60, axis=1)
    running = power > 0
    paths = MinutePaths(ORIGIN, power, running, np.zeros_like(running))
    grid = regular_grid(ORIGIN, minutes=15, hours=24)
    forecast = HvacForecast(
        ORIGIN,
        grid,
        interval_stats(paths, grid),
        paths,
        power[:16],
        0,
        InputTier.T1_POWER,
        HvacFamily.FIXED_SPEED,
        None,
        {},
        "fixture",
    )
    actual = pd.DataFrame(
        {"power_w": 1000.0},
        index=pd.date_range(
            pd.Timestamp(ORIGIN) - pd.Timedelta(minutes=1), periods=1441, freq="min"
        ),
    )
    expected = np.full(96, 0.25)
    baseline = BaselineForecast(
        expected, np.repeat(expected[None, :], 3, axis=0), np.full(96, 1000.0), {}
    )
    return forecast, actual, {"B1": baseline}


def test_full_ensemble_scores_and_b5_pairs_never_use_display_paths(evidence):
    forecast, actual, baselines = evidence
    energy = quarter_scores(forecast, actual, baselines, threshold_w=200, hours=24)
    cycle = energy.loc[energy.model.eq("cycle")]
    assert cycle.expected_kwh.to_numpy() == pytest.approx(np.full(96, 0.25))
    assert cycle.crps.to_numpy() == pytest.approx(np.full(96, 0.125))
    assert cycle.pinball.to_numpy() == pytest.approx(np.full(96, 1 / 60))
    b5 = np.full(12, 0.25)
    b5[2] = np.nan
    runs = run_scores(forecast, actual, threshold_w=200, hours=24, r1c=b5)
    near = runs.loc[runs.model.eq("cycle") & runs.minutes.eq(5)]
    assert near.probability.to_numpy() == pytest.approx(np.full(24, 0.5))
    assert near.brier.to_numpy() == pytest.approx(np.full(24, 0.25))
    comparison = runs.loc[runs.model.eq("B5")]
    assert comparison.eligible.sum() == 11
    assert comparison.brier.dropna().to_numpy() == pytest.approx(np.full(11, 0.5625))


def test_missing_minute_keeps_a4_unscorable_but_reports_exact_observed_footprint(
    evidence,
):
    forecast, actual, baselines = evidence
    actual.loc[pd.Timestamp(ORIGIN) + pd.Timedelta(minutes=2), "power_w"] = np.nan
    rows = energy_scores(forecast, actual, baselines, hours=24)
    day = rows.loc[rows.duration_h.eq(24)]
    assert not day.eligible.any()
    assert day.error_kwh.isna().all() and day.actual_kwh.isna().all()
    assert day.observed_minutes.tolist() == [1439, 1439]
    assert day.coverage.to_numpy() == pytest.approx(np.full(2, 1439 / 1440))
    assert day.partial_actual_kwh.to_numpy() == pytest.approx(np.full(2, 1439 / 60))
    assert day.partial_error_kwh.to_numpy() == pytest.approx(np.zeros(2))
    assert day.partial_method.tolist() == [
        "minute_path_mean",
        "quarter_mean_piecewise_constant",
    ]
    cadence = cadence_scores(forecast, actual, threshold_w=200, hours=24)
    assert not cadence.eligible.iloc[0]
    assert cadence.loc[cadence.duration_h.eq(24), "run_length_ks"].isna().all()
    assert cadence.loc[cadence.duration_h.eq(6)].eligible.sum() == 3


def test_t0_state_is_unknown_and_boundary_start_needs_observed_prior_minute(evidence):
    forecast, actual, _ = evidence
    paths = MinutePaths(ORIGIN, forecast.minute_paths.power_w, None)
    t0 = replace(
        forecast, minute_paths=paths, intervals=interval_stats(paths, forecast.grid)
    )
    runs = run_scores(t0, actual, threshold_w=None, hours=24)
    assert not runs.eligible.any() and runs.probability.isna().all()
    assert not cadence_scores(t0, actual, threshold_w=None, hours=24).eligible.any()
    actual.iloc[0, 0] = np.nan
    cadence = cadence_scores(forecast, actual, threshold_w=200, hours=24)
    assert not cadence.eligible.iloc[0]
    assert cadence.eligible.iloc[1]


def test_blocks_use_shared_directional_math_and_exclude_whole_unknown_blocks(evidence):
    forecast, actual, baselines = evidence
    grid = regular_grid(ORIGIN, minutes=15, hours=1)
    windows = [
        SettlementWindow(step.start, step.end, "a" if i < 2 else "b", 100.0, None)
        for i, step in enumerate(grid)
    ]
    index = actual.index
    site = pd.DataFrame({"base_w": 0.0, "pv_w": 0.0}, index=index)
    valid = pd.Series(True, index=index)
    result = block_scores(forecast, actual, site, valid, windows, baselines)
    cycle = result.loc[result.model.eq("cycle")]
    assert cycle.reserve_q90_wh.tolist() == pytest.approx([800, 800])
    assert cycle.actual_need_wh.tolist() == pytest.approx([300, 300])
    assert cycle.pinball_q90.tolist() == pytest.approx([50, 50])
    assert cycle.peak_crps_kwh.tolist() == pytest.approx([0.125, 0.125])
    valid.loc[pd.Timestamp(ORIGIN) + pd.Timedelta(minutes=2)] = False
    result = block_scores(forecast, actual, site, valid, windows, baselines)
    first = result.loc[result.block_id.eq("a")]
    assert not first.eligible.any() and first.actual_need_wh.isna().all()
    assert first.loc[first.model.eq("cycle"), "peak_crps_kwh"].iloc[0] == pytest.approx(
        0.125
    )
    assert result.loc[result.block_id.eq("b"), "eligible"].all()
