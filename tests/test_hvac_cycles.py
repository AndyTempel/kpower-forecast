"""Exact censoring, gap, defrost and intervention semantics on hand-built traces."""

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.cycles import extract_cycles
from kpower_forecast.hvac.types import HvacModelConfig


@pytest.fixture
def config():
    return HvacModelConfig(model_id="test", timezone="UTC")


@pytest.fixture
def trace():
    index = pd.date_range("2026-11-09T00:00Z", periods=100, freq="min")
    running = np.zeros(100)
    running[:5] = 1
    running[25:35] = 1
    running[40:50] = 1
    running[70:80] = 1
    frame = pd.DataFrame(
        {
            "running": running,
            "power_w": np.where(running > 0, 4500.0, 60.0),
            "outdoor_c": 5.0,
            "sun": 0.0,
            "supply_c": 35.0,
            "defrost": 0.0,
        },
        index=index,
    )
    frame.loc[index[35:40], "defrost"] = 1.0
    events = pd.DataFrame(
        {
            "time": [index[0], index[79]],
            "mode": ["NORMAL", "BOOST"],
            "source": ["plan", "manual"],
        }
    )
    return frame, events


def test_defrost_merge_censoring_switches_and_unknown_window_left_edge(config, trace):
    frame, events = trace
    result = extract_cycles(frame, events, config)
    spells = result.sojourns
    assert result.dropped_left_edges == 1
    assert spells.kind.tolist() == ["idle", "run", "idle", "run", "idle"]
    assert spells.minutes.tolist() == [20.0, 25.0, 20.0, 10.0, 20.0]
    assert spells.end_cause.tolist() == [
        "natural",
        "natural",
        "natural",
        "command",
        "window",
    ]
    merged = spells.iloc[1]
    assert merged.defrost_interruptions == 1 and merged.defrost_minutes == 5
    assert merged.power_w == 4500
    switch = result.mode_switches.iloc[0]
    assert (
        switch.from_mode,
        switch.to_mode,
        switch.elapsed_min,
        switch.outcome_min,
    ) == ("NORMAL", "BOOST", 9, 1)
    assert switch.outcome_censored


@pytest.mark.parametrize("gap_minutes,expected_gap", [(5, False), (6, True)])
def test_gap_tolerance_never_fills_missing_states(
    config, trace, gap_minutes, expected_gap
):
    frame, events = trace
    frame = frame.drop(frame.index[28 : 28 + gap_minutes - 1])
    result = extract_cycles(frame, events, config)
    assert result.sojourns.end_cause.eq("gap").any() == expected_gap
    assert result.dropped_left_edges == 1 + int(expected_gap)


def test_mode_command_in_defrost_gap_prevents_merge_and_offset_end_is_censored(
    config, trace
):
    frame, events = trace
    extra = pd.DataFrame(
        {"time": [frame.index[37]], "mode": ["NORMAL"], "source": ["protection"]}
    )
    events = pd.concat([events, extra], ignore_index=True)
    controls = pd.DataFrame(
        {"time": [frame.index[50]], "metric": ["RETURN_TEMPERATURE_OFFSET"]}
    )
    result = extract_cycles(frame, events, config, control_events=controls)
    runs = result.sojourns.loc[result.sojourns.kind.eq("run")]
    assert runs.minutes.tolist() == [10.0, 10.0, 10.0]
    assert runs.end_cause.tolist() == ["command", "command", "command"]
    assert runs.censored.all()


def test_first_recovery_run_remains_marked_after_intervening_idle(config):
    index = pd.date_range("2026-11-09T00:00Z", periods=30, freq="min")
    state = np.r_[np.zeros(10), np.ones(10), np.zeros(10)]
    frame = pd.DataFrame(
        {"running": state, "power_w": np.where(state > 0, 4500.0, 60.0)}, index=index
    )
    events = pd.DataFrame({"time": [index[0], index[5]], "mode": ["OFF", "NORMAL"]})
    result = extract_cycles(frame, events, config)
    assert result.sojourns.loc[
        result.sojourns.kind.eq("run"), "first_after_off"
    ].tolist() == [True]
