"""Eligibility, calibration and numerical score contracts, independent of models."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.evaluation import (
    actual_intervals,
    calibration_slope,
    ks_distance,
    lead_bucket,
    mixed_grid,
    pinball,
    regular_grid,
    reliability_bins,
)

ORIGIN = datetime(2026, 11, 9, tzinfo=UTC)


def test_unknown_actual_minute_excludes_only_affected_intervals_without_filling():
    frame = pd.DataFrame(
        {"power_w": 1000.0}, index=pd.date_range(ORIGIN, periods=60, freq="min")
    )
    frame.iloc[2, 0] = np.nan
    actual = actual_intervals(
        frame, regular_grid(ORIGIN, minutes=15, hours=1), threshold_w=200.0
    )
    assert np.isnan(actual.energy_kwh[0]) and np.isnan(actual.ran[0])
    np.testing.assert_array_equal(actual.eligible_energy, [False, True, True, True])
    np.testing.assert_allclose(actual.energy_kwh[1:], [0.25, 0.25, 0.25])
    assert np.isnan(frame.power_w.iloc[2])
    assert not actual.energy_kwh.flags.writeable
    unknown_state = actual_intervals(
        frame, regular_grid(ORIGIN, minutes=15, hours=1), threshold_w=None
    )
    assert np.isnan(unknown_state.ran).all()
    np.testing.assert_allclose(unknown_state.energy_kwh[1:], [0.25, 0.25, 0.25])


def test_scores_use_raw_probabilities_and_registered_grids():
    assert pinball(
        np.array([1.0, 0.0]), np.array([[0.5, 1.0], [1.0, 1.0], [2.0, 1.0]])
    ) == pytest.approx([0.05, 0.5])
    bins = reliability_bins(np.array([0.0, 1.0, 0.25]), np.array([0.0, 1.0, 1.0]))
    assert bins[0]["predicted"] == 0.0 and bins[9]["predicted"] == 1.0
    assert bins[2]["n"] == 1 and bins[2]["observed"] == 1.0
    assert bins[1]["n"] == 0 and bins[1]["predicted"] is None
    grid = mixed_grid(ORIGIN)
    assert len(grid) == 24 + 70 * 4
    assert grid[23].minutes == 5 and grid[24].minutes == 15
    assert grid[-1].end == ORIGIN + pd.Timedelta(hours=72)
    assert lead_bucket(15) == "15-30m" and lead_bucket(60) == "1-3h"
    assert lead_bucket(72 * 60) is None
    assert ks_distance(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert ks_distance(np.array([]), np.array([1.0])) is None


def test_logistic_calibration_recovers_slope_and_reports_unidentifiable_cases():
    rng = np.random.default_rng(77)
    probability = rng.uniform(0.02, 0.98, 20000)
    actual = (rng.random(len(probability)) < probability).astype(float)
    fit = calibration_slope(probability, actual, n_paths=256)
    assert fit.reason is None
    assert fit.slope == pytest.approx(1.0, abs=0.08)
    assert fit.intercept == pytest.approx(0.0, abs=0.08)
    assert (
        calibration_slope(
            np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0]), n_paths=256
        ).reason
        == "separated_or_singular"
    )
    assert (
        calibration_slope(np.full(10, 0.5), np.tile([0.0, 1.0], 5), n_paths=256).reason
        == "constant_probability"
    )
    assert (
        calibration_slope(probability, np.zeros(len(probability)), n_paths=256).reason
        == "insufficient_outcome_variation"
    )
    with pytest.raises(ValueError, match="invalid probability"):
        calibration_slope(np.array([np.nan]), np.array([1.0]), n_paths=256)
