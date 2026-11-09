"""Recover a known hysteresis process before using its fitted parameters."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.cycle_model import fit_cycles
from kpower_forecast.hvac.types import HvacCapabilities, HvacModelConfig


@pytest.fixture(scope="module")
def physical_process():
    rng = np.random.default_rng(201)
    n = 2400
    temperature = rng.uniform(0, 15, n)
    sun = rng.uniform(0, 0.5, n)
    hour = rng.uniform(0, 2 * np.pi, n)
    idle_rate = (
        0.002
        * (20 - temperature)
        * np.exp(-0.25 * sun + 0.04 * np.sin(hour) - 0.02 * np.cos(hour))
    )
    total_rate = 0.11 * (1 + 0.01 * (temperature - 7))
    frames = []
    for kind, median in [
        ("idle", 1 / idle_rate),
        ("run", 1 / (total_rate - idle_rate)),
    ]:
        duration = median * np.exp(0.18 * rng.normal(size=n))
        cutoff = np.quantile(duration, 0.8)
        frames.append(
            pd.DataFrame(
                {
                    "kind": kind,
                    "outdoor_c": temperature,
                    "sun": sun,
                    "hour_sin": np.sin(hour),
                    "hour_cos": np.cos(hour),
                    "mode": "NORMAL",
                    "minutes": np.minimum(duration, cutoff),
                    "censored": duration > cutoff,
                    "start": pd.date_range(
                        datetime(2026, 10, 1, tzinfo=timezone.utc),
                        periods=n,
                        freq="15min",
                    ),
                    "first_after_off": False,
                    "defrost_minutes": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def physical_fit(physical_process):
    cfg = HvacModelConfig(model_id="synthetic", timezone="UTC", n_parameter_sets=1)
    fit, reason = fit_cycles(
        physical_process,
        cfg,
        HvacCapabilities(
            family="fixed_speed",
            fingerprint="rated",
            rated_points=[
                {
                    "source_temp_c": 0,
                    "supply_temp_c": 35,
                    "heat_output_kw": 9.3,
                    "electrical_input_kw": 3,
                },
                {
                    "source_temp_c": 7,
                    "supply_temp_c": 35,
                    "heat_output_kw": 10,
                    "electrical_input_kw": 3,
                },
            ],
        ),
    )
    assert reason is None and fit is not None
    return fit


def test_physical_fit_recovers_balance_and_cycle_medians(physical_fit):
    model = physical_fit.parameter_sets[0]
    assert abs(model.t_bal_c - 20) <= 1
    assert not physical_fit.t_bal_at_grid_edge
    t = np.array([0.0, 7.0, 15.0])
    idle, run, _ = model.locations(t, "NORMAL", sun=0.25)
    a = 0.002 * (20 - t) * np.exp(-0.25 * 0.25)
    r = 0.11 * (1 + 0.01 * (t - 7))
    np.testing.assert_allclose(np.exp(idle), 1 / a, rtol=0.1)
    np.testing.assert_allclose(np.exp(run), 1 / (r - a), rtol=0.1)
    assert model.capacity_slope == pytest.approx(0.01)
    cold = np.array([15.0, 7.0, 0.0, -10.0, -40.0])
    duty, _ = model.stationary(cold, "NORMAL")
    mi, mr, over = model.locations(cold, "NORMAL")
    assert np.all(np.diff(duty) >= 0) and np.all(duty <= 1)
    assert np.all(np.diff(np.exp(mi)) < 0)
    assert np.all(np.diff(np.exp(mr[:4])) > 0) and over[-1]


def test_missing_reference_cycles_and_narrow_weather_are_explicit_fallbacks(
    physical_process,
):
    cfg = HvacModelConfig(model_id="test", timezone="UTC", n_parameter_sets=1)
    cap = HvacCapabilities(family="fixed_speed", fingerprint="unit")
    assert fit_cycles(physical_process.iloc[:20], cfg, cap)[1] == "insufficient_cycles"
    narrow = physical_process.copy()
    narrow["outdoor_c"] = 5.0
    assert fit_cycles(narrow, cfg, cap)[1] == "narrow_outdoor_range"


def test_total_bootstrap_sets_and_day_resampling_are_deterministic(physical_process):
    cfg = HvacModelConfig(
        model_id="bootstrap", timezone="UTC", n_parameter_sets=3, seed=17
    )
    cap = HvacCapabilities(
        family="fixed_speed",
        fingerprint="rated",
        rated_points=[
            {
                "source_temp_c": 0,
                "supply_temp_c": 35,
                "heat_output_kw": 9.3,
                "electrical_input_kw": 3,
            },
            {
                "source_temp_c": 7,
                "supply_temp_c": 35,
                "heat_output_kw": 10,
                "electrical_input_kw": 3,
            },
        ],
    )
    first, reason = fit_cycles(physical_process, cfg, cap)
    second, _ = fit_cycles(physical_process, cfg, cap)
    assert reason is None and first is not None and second is not None
    assert len(first.parameter_sets) == 3
    assert first.parameter_sets == second.parameter_sets
    assert first.parameter_sets[0] != first.parameter_sets[1]


def test_defrost_is_fitted_as_active_run_time_and_switch_effects_use_censoring(
    physical_process, physical_fit
):
    modified = physical_process.copy()
    on = modified.kind.eq("run")
    modified.loc[on, "minutes"] += 5
    modified.loc[on, "defrost_minutes"] = 5
    modified["defrost_interruptions"] = on.astype(int)
    rng = np.random.default_rng(33)
    outcome = np.exp(np.log(8.0) + 0.2 * rng.normal(size=80))
    limit = np.quantile(outcome, 0.7)
    switches = pd.DataFrame(
        {
            "from_mode": "ECO",
            "to_mode": "NORMAL",
            "state": "idle",
            "outcome_min": np.minimum(outcome, limit),
            "outcome_censored": outcome > limit,
        }
    )
    cfg = HvacModelConfig(model_id="active", timezone="UTC", n_parameter_sets=1)
    cap = HvacCapabilities(
        family="fixed_speed",
        fingerprint="rated",
        rated_points=[
            {
                "source_temp_c": 0,
                "supply_temp_c": 35,
                "heat_output_kw": 9.3,
                "electrical_input_kw": 3,
            },
            {
                "source_temp_c": 7,
                "supply_temp_c": 35,
                "heat_output_kw": 10,
                "electrical_input_kw": 3,
            },
        ],
    )
    fit, reason = fit_cycles(modified, cfg, cap, mode_switches=switches)
    assert reason is None and fit is not None
    np.testing.assert_allclose(
        fit.parameter_sets[0].idle_beta,
        physical_fit.parameter_sets[0].idle_beta,
        atol=1e-7,
    )
    assert fit.parameter_sets[0].rho == pytest.approx(
        physical_fit.parameter_sets[0].rho, rel=1e-7
    )
    np.testing.assert_allclose(fit.defrost_rate, [1, 1, 1])
    assert fit.defrost_length_min == 5.0
    assert len(fit.switch_effects) == 1 and fit.switch_effects[0].mu == pytest.approx(
        np.log(8.0), abs=0.1
    )
