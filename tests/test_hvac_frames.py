"""Minute availability, outdoor grace and same-policy eligibility regressions."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.frames import (
    history_verdict,
    history_window,
    outdoor_frame,
    prepare_minutes,
    running_states,
)
from kpower_forecast.hvac.types import HistoryPolicy, HvacModelConfig


@pytest.fixture
def config():
    return HvacModelConfig(
        model_id="test", timezone="Europe/Ljubljana", ingest_lag_min=2
    )


@pytest.fixture
def minutes():
    index = pd.date_range("2026-11-09T00:00Z", periods=90, freq="min")
    return pd.DataFrame(
        {
            "power_w": np.tile([60.0, 4500.0, 4500.0], 30),
            "power_w__measured_valid": True,
            "power_w__quality": "good",
        },
        index=index,
    )


def test_availability_unknown_quality_and_bridged_power_are_never_filled(
    config, minutes
):
    origin = minutes.index[20].to_pydatetime()
    minutes["power_w__measured_valid"] = minutes["power_w__measured_valid"].astype(
        "boolean"
    )
    minutes.loc[minutes.index[1], "power_w__measured_valid"] = False
    minutes.loc[minutes.index[2], "power_w__measured_valid"] = pd.NA
    minutes["power_w__bridged"] = False
    minutes.loc[minutes.index[3], "power_w__bridged"] = True
    result = prepare_minutes(minutes, config, as_of=origin, training=False)
    assert result.index[-1] == minutes.index[17]
    # Explicitly invalid and bridged minutes are missing; an untagged (unknown)
    # minute keeps its observed value, as aggregated history has no tags.
    assert result.power_w.iloc[0] == 60
    assert np.isnan(result.power_w.iloc[1]) and np.isnan(result.power_w.iloc[3])
    assert result.power_w.iloc[2] == minutes.power_w.iloc[2]
    clean = prepare_minutes(minutes.iloc[:18], config, as_of=origin, training=False)
    pd.testing.assert_frame_equal(result, clean)


def test_outdoor_precedence_hold_and_first_invalidation_grace(config, minutes):
    frame = minutes.copy()
    frame["outdoor_env_c"] = np.nan
    frame["outdoor_unit_c"] = np.nan
    frame.loc[frame.index[0], "outdoor_env_c"] = 5.0
    frame.loc[frame.index[2], "outdoor_unit_c"] = 8.0
    frame["outdoor_env_invalid"] = 0.0
    frame.loc[frame.index[3], "outdoor_env_invalid"] = 1.0
    weather = pd.DataFrame(
        {"temperature_2m": [2.0, 2.0, 2.0], "shortwave_radiation": [0.0, 100.0, 0.0]},
        index=pd.date_range(frame.index[0], periods=3, freq="h"),
    )
    result = outdoor_frame(frame, weather, config)
    assert result.outdoor_c.iloc[0] == 5 and result.outdoor_c.iloc[2] == 8
    assert result.outdoor_c.iloc[62] == 8 and result.outdoor_c.iloc[63] == 2
    frame["outdoor_unit_c"] = np.nan
    result = outdoor_frame(frame, weather, config)
    assert result.outdoor_c.iloc[17] == 5  # 14 minutes since explicit invalidation
    assert result.outdoor_c.iloc[19] == 2  # 16 minutes since invalidation
    assert result.outdoor_source.iloc[19] == "weather"
    frame["outdoor_env_invalid"] = 0.0
    result = outdoor_frame(frame, weather, config)
    assert result.outdoor_c.iloc[60] == 5 and result.outdoor_c.iloc[61] == 2


def test_histogram_guard_and_unit_flag_do_not_turn_unknown_power_into_idle(minutes):
    result = running_states(minutes)
    assert result.reason is None and 200 <= result.threshold_w <= 2250
    frame = minutes.copy()
    frame["running"] = 0.0
    frame.loc[frame.index[3], "power_w"] = np.nan
    result = running_states(frame, 1000.0)
    assert result.running.iloc[1] == 0 and pd.isna(result.running.iloc[3])
    flat = minutes.copy()
    flat["power_w"] = 1500.0
    assert running_states(flat).reason == "not_cycling"


@pytest.mark.parametrize(
    "case_name",
    [
        "good",
        "short_gap",
        "large_gap",
        "missing_coverage",
        "invalid_target",
        "optional_sensor_outage",
        "recent_change",
    ],
)
def test_history_policy_fixture(case_name, config):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/hvac_history_policy.json").read_text()
    )
    case = fixture["cases"][case_name]
    index = pd.date_range(fixture["start"], periods=fixture["periods"], freq="min")
    frame = pd.DataFrame({"power_w": 1000.0, "coverage_ratio": 1.0}, index=index)
    for a, b in case["missing_ranges"]:
        frame.iloc[a:b, 0] = np.nan
    if case_name == "missing_coverage":
        frame = frame.drop(columns="coverage_ratio")
    if case_name == "optional_sensor_outage":
        frame["supply_c"] = np.nan
    policy = HistoryPolicy(
        version=fixture["version"],
        settings=fixture["settings"],
        scope=case["scope"],
        measured_valid_qualities=fixture["measured_valid_qualities"],
        last_fingerprint_change_at=case["last_change"],
    )
    selected = history_window(frame, policy)
    verdict = history_verdict(
        selected, policy, as_of=datetime.fromisoformat(fixture["as_of"])
    )
    expected = case["expected"]
    assert len(selected) == expected["rows"]
    assert (selected.index[0].isoformat() if len(selected) else None) == expected[
        "start"
    ]
    assert verdict.span_days == pytest.approx(expected["span_days"])
    assert verdict.observed_density == pytest.approx(expected["density"])
    assert (
        verdict.reliable == expected["reliable"]
        and verdict.reason == expected["reason"]
    )
    if case["scope"] == "system_attributed_heating":
        assert verdict.full_row_fraction == pytest.approx(expected["full_fraction"])
        assert verdict.mean_coverage == pytest.approx(expected["mean_coverage"])
