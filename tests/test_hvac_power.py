"""Meter watts, standby and observed ramps are learned without invented zeros."""

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.power import fit_power
from kpower_forecast.hvac.types import HvacModelConfig, MeteredScope


@pytest.fixture
def power_trace():
    n = 80
    index = pd.date_range("2026-11-01T00:00Z", periods=n * 15 + 5, freq="min")
    frame = pd.DataFrame({"power_w": 60.0, "running": 0.0}, index=index)
    frame.iloc[:5, 0] = 25.0
    rows = []
    for i in range(n):
        start = 5 + 15 * i
        temperature = 15 * i / (n - 1)
        supply = 35 + 1.3 * np.sin(0.31 * i)
        level = 4000 - 20 * (temperature - 7) + 30 * (supply - 35)
        frame.iloc[start : start + 10, 0] = level
        frame.iloc[start : start + 10, 1] = 1.0
        frame.iloc[start, 0] = 0.5 * level
        frame.iloc[start + 1, 0] = 0.8 * level
        rows.append(
            {
                "kind": "run",
                "start": index[start],
                "end": index[start + 10],
                "power_w": level,
                "outdoor_c": temperature,
                "supply_c": supply,
                "censored": False,
            }
        )
    runs = pd.DataFrame(rows)
    runs.loc[0, "power_w"] = 1e6
    runs.loc[0, "censored"] = True
    events = pd.DataFrame({"time": [index[0], index[5]], "mode": ["OFF", "NORMAL"]})
    return frame, runs, events


def test_power_regression_resists_an_outlier_and_retains_observed_standby_ramp(
    power_trace,
):
    frame, runs, events = power_trace
    cfg = HvacModelConfig(model_id="power", timezone="UTC")
    result = fit_power(frame, runs, events, cfg, MeteredScope.WHOLE_UNIT)
    np.testing.assert_allclose(result.coefficients, [4000.0, -20.0, 30.0], atol=1e-5)
    np.testing.assert_allclose(result.start_ramp, [0.5, 0.8, 1.0], atol=1e-8)
    assert result.idle_default_w == 60 and result.off_w == 25
    assert result.defrost_w is None and result.metered_scope is MeteredScope.WHOLE_UNIT
    assert result.run_power(7.0, 35.0) == pytest.approx(4000.0)


def test_control_setpoint_is_a_recorded_alternative_not_mixed_with_supply(power_trace):
    frame, runs, events = power_trace
    setpoint = runs.rename(columns={"supply_c": "control_setpoint_c"})
    result = fit_power(
        frame,
        setpoint,
        events,
        HvacModelConfig(model_id="power", timezone="UTC"),
        MeteredScope.WHOLE_UNIT,
    )
    assert result.uses_supply and result.temperature_source == "control_setpoint_c"
    np.testing.assert_allclose(result.coefficients, [4000.0, -20.0, 30.0], atol=1e-5)
    assert result.supply_reference_c == pytest.approx(
        setpoint.control_setpoint_c.mean()
    )
    # A supported measured supply series takes precedence; sources are not mixed.
    both = runs.copy()
    both["control_setpoint_c"] = 50.0
    supply = fit_power(
        frame,
        both,
        events,
        HvacModelConfig(model_id="power", timezone="UTC"),
        MeteredScope.WHOLE_UNIT,
    )
    assert supply.temperature_source == "supply_c"
    np.testing.assert_allclose(supply.coefficients, [4000.0, -20.0, 30.0], atol=1e-5)
