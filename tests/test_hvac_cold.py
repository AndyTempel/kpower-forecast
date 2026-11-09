"""Hard cold-extrapolation shape and overload gate on a warm-trained model."""

import numpy as np
import pytest

from kpower_forecast.hvac.cycle_model import CycleParameters


def test_cold_shape_has_monotone_duty_and_no_simulation_run_rate_floor():
    # Known physical parameters, warm-training recovery is separately gated in
    # test_hvac_cycle_model. This exercises the same fitted evaluator at extremes.
    model = CycleParameters(
        20.0, (), (-np.log(0.002), 0.0, 0.0, 0.0), 0.2, 0.11, 0.01, (), (), 0.2
    )
    t = np.array([15.0, 7.0, 0.0, -10.0, -40.0])
    idle, run, over = model.locations(t, "NORMAL")
    duty, touch = model.stationary(t, "NORMAL")
    assert np.all(np.diff(duty) >= 0) and np.all(duty <= 1)
    assert np.all(np.diff(np.exp(idle)) < 0)
    assert np.all(np.diff(np.exp(run[:4])) > 0)
    assert over[-1] and np.isinf(run[-1]) and duty[-1] == 1 and touch[-1] == 1
    _, r, _ = model.rates(np.array([0.0, 7.0]), "NORMAL")
    assert r[0] / r[1] == pytest.approx(0.93)


def test_touch_probability_uses_duration_means_and_partial_idle_expectation():
    model = CycleParameters(
        20.0, (), (-np.log(1 / 585), 0.0, 0.0, 0.0), 0.001, 4 / 45, 0.0, (), (), 0.001
    )
    # At 7C: a=1/45, b=1/15. In the deterministic limit D=.25,
    # but a fifteen-minute interval touches a run half the time.
    duty, touch = model.stationary(7.0, "NORMAL", interval_min=15.0)
    assert duty == pytest.approx(0.25, abs=1e-5)
    assert touch == pytest.approx(0.5, abs=1e-5)
