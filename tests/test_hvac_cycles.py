"""Spell extraction: cleaning, command censoring and unknown edges."""

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.cycles import clean_running, extract_cycles
from kpower_forecast.hvac.types import HvacModelConfig

START = pd.Timestamp("2026-11-09T00:00Z")


@pytest.fixture
def config():
    return HvacModelConfig(model_id="t", timezone="UTC")


def frame(states):
    index = pd.date_range(START, periods=len(states), freq="min")
    return pd.DataFrame(
        {"running": np.asarray(states, float), "outdoor_c": 5.0, "sun": 0.0},
        index=index,
    )


def events(*rows):
    return pd.DataFrame(
        [(START + pd.Timedelta(minutes=m), mode, "plan") for m, mode in rows],
        columns=["time", "mode", "source"],
    )


def test_clean_merges_short_dips_and_drops_blips_but_not_at_gaps():
    nan = np.nan
    raw = pd.Series([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, nan, 1, 0, 0])
    cleaned = clean_running(raw, max_dip_min=3, min_run_min=3).to_numpy()
    assert list(cleaned[1:6]) == [1, 1, 1, 1, 1]  # one-minute dip merged
    assert cleaned[10] == 0  # one-minute blip between idles removed
    assert cleaned[15] == 1 and np.isnan(cleaned[16])  # touching a gap: unknown
    assert cleaned[17] == 1  # blip right after a gap is left alone


def test_short_gap_inside_one_state_is_bridged_but_not_between_states():
    nan = np.nan
    raw = pd.Series([1, 0, 0, nan, nan, 0, 0, 1, nan, 0, 0, nan] + [nan] * 6 + [0])
    cleaned = clean_running(raw, max_dip_min=0, min_run_min=1, max_gap_min=5)
    v = cleaned.to_numpy()
    assert list(v[1:7]) == [0, 0, 0, 0, 0, 0]  # idle continues through 2 min
    assert np.isnan(v[8])  # run -> idle boundary unknown: not bridged
    assert np.isnan(v[11:18]).all()  # seven missing minutes exceed the tolerance


def test_natural_spells_previous_run_and_unknown_left_edge(config):
    states = [1] * 5 + [0] * 40 + [1] * 15 + [0] * 45 + [1] * 3
    s = extract_cycles(frame(states), events((0, "NORMAL")), config).sojourns
    # The first run starts at the window edge: its true start is unknown.
    assert list(s.kind) == ["idle", "run", "idle", "run"]
    assert list(s.minutes) == [40, 15, 45, 3]
    assert list(s.censored) == [False, False, False, True]
    assert s.prev_run_min.iloc[2] == 15
    assert np.isnan(s.prev_run_min.iloc[0])  # preceding run had an unknown start
    assert s.end_cause.iloc[-1] == "window"


def test_real_mode_change_censors_but_identical_resend_does_not(config):
    states = [0] * 10 + [1] * 20 + [0] * 50 + [1] * 10
    ev = events((0, "NORMAL"), (15, "NORMAL"), (40, "ECO"), (79, "NORMAL"))
    s = extract_cycles(frame(states), ev, config).sojourns.set_index("start")
    run = s.loc[START + pd.Timedelta(minutes=10)]
    assert not run.censored and run.minutes == 20  # resend at minute 15 ignored
    idle = s.loc[START + pd.Timedelta(minutes=30)]
    # ECO at minute 40 truncates the idle there; the rest has no known start.
    assert idle.censored and idle.minutes == 10 and idle.end_cause == "command"
    assert len(s) == 3


def test_command_just_before_a_stop_censors_the_spell(config):
    states = [0] * 10 + [1] * 12 + [0] * 30 + [1] * 5
    ev = events((0, "NORMAL"), (21, "ECO"))
    s = extract_cycles(frame(states), ev, config).sojourns
    run = s.loc[s.kind == "run"].iloc[0]
    assert run.censored and run.minutes == 12 and run.end_cause == "command"


def test_gaps_censor_and_off_idle_is_not_a_thermal_spell(config):
    states = [0] * 5 + [1] * 10 + [np.nan] * 3 + [1] * 5 + [0] * 30 + [1] * 5
    s = extract_cycles(frame(states), events((0, "NORMAL")), config).sojourns
    assert s.iloc[0].kind == "run" and s.iloc[0].censored
    assert s.iloc[0].end_cause == "gap"
    off = [0] * 5 + [1] * 10 + [0] * 60 + [1] * 10
    s = extract_cycles(frame(off), events((0, "NORMAL"), (15, "OFF")), config).sojourns
    assert not ((s.kind == "idle") & (s["mode"] == "OFF")).any()
