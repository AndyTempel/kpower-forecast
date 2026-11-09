"""One bundle availability gateway for fitting, prediction and separated outcomes."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from kpower_forecast.hvac.bundle import (
    BundleManifest,
    HvacEvaluationBundle,
    _origin_weather,
)

T0 = datetime(2026, 11, 9, 12, tzinfo=UTC)


@pytest.fixture
def bundle():
    owner = json.loads(
        (Path(__file__).parent / "fixtures/hvac_owner_policy.json").read_text()
    )
    schedule = json.loads(
        (Path(__file__).parent / "fixtures/hvac_schedule_parity.json").read_text()
    )["cases"][0]["payload"]
    canonical = json.dumps(
        {k: v for k, v in schedule.items() if k not in {"generated_at", "valid_until"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    index = pd.date_range(T0 - timedelta(minutes=5), periods=7, freq="min")
    minutes = pd.DataFrame(
        {
            "power_w": 1000.0,
            "power_w__measured_valid": True,
            "power_w__quality": "good",
            "power_w__source_device": ["hp"] * 3 + ["future-source"] * 4,
        },
        index=index,
    )
    events = pd.DataFrame(
        {
            "time": [T0 - timedelta(minutes=4), T0 + timedelta(minutes=1)],
            "mode": ["NORMAL", "OFF"],
            "source": ["manual", "thermostat"],
        }
    )
    weather = pd.DataFrame(
        {
            "ds": pd.date_range(T0 - timedelta(hours=1), periods=3, freq="h"),
            "temperature_2m": 7.0,
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 2.0,
            "weather_source": "historical_forecast",
        }
    )
    vintages = pd.DataFrame(
        {
            "valid_at": [
                T0,
                T0 + timedelta(hours=1),
                T0 + timedelta(hours=1),
                T0 + timedelta(hours=1),
            ],
            "available_at": [
                T0 - timedelta(days=1),
                T0 - timedelta(days=1) + timedelta(hours=1),
                T0 - timedelta(minutes=1),
                T0 + timedelta(minutes=1),
            ],
            "vintage_kind": [
                "approx_previous_runs",
                "approx_previous_runs",
                "archived",
                "archived",
            ],
            "temperature_2m": [7.0, 7.0, 8.0, 99.0],
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 2.0,
        }
    )
    schedules = pd.DataFrame(
        {
            "available_at": [T0 - timedelta(minutes=1), T0 + timedelta(minutes=5)],
            "generated_at": [T0 - timedelta(minutes=1), T0 + timedelta(minutes=4)],
            "valid_until": [T0 + timedelta(minutes=14), T0 + timedelta(minutes=19)],
            "payload_json": [json.dumps(schedule)] * 2,
            "content_hash": [digest] * 2,
        }
    )
    tables = {
        "minutes.parquet": minutes,
        "mode_events.parquet": events,
        "weather.parquet": weather,
        "weather_vintages.parquet": vintages,
        "schedules.parquet": schedules,
        "ledger.parquet": pd.DataFrame(
            {"observed_until": [T0 - timedelta(minutes=1), T0 + timedelta(minutes=1)]}
        ),
        "served_forecast.parquet": pd.DataFrame(
            {
                "origin": [T0 - timedelta(minutes=1), T0 + timedelta(minutes=1)],
                "expected_w": [1000.0, 99999.0],
            }
        ),
    }
    documents = {
        "capabilities.json": {"family": "fixed_speed", "fingerprint": "synthetic"},
        "history_policy.json": {
            "minute_history_policy": owner["settings"],
            "measured_valid_qualities": owner["measured_valid_qualities"],
        },
        "minute_qualification.json": {"scope": "direct_device_minutes"},
    }
    manifest = BundleManifest(
        exported_at=T0 + timedelta(minutes=5),
        start=index[0],
        end=T0 + timedelta(minutes=2),
        controller_id="controller",
        device_id="hp",
        timezone="UTC",
        latitude=0.0,
        longitude=0.0,
        ems_commit="synthetic",
        ingest_lag_min=2,
        training_ready=True,
        files={},
    )
    return HvacEvaluationBundle(manifest, tables, documents)


def test_gateway_excludes_delayed_features_and_separates_oracle_outcomes(bundle):
    view = bundle.available_at(T0, include_inputs=True)
    assert view.training.minutes.index[-1] == T0 - timedelta(minutes=3)
    assert view.training.lineage["power_w__source_device"] == "hp"
    assert view.last_mode == "NORMAL" and view.last_mode_event is None
    assert len(view.training.mode_events) == 1
    assert view.training.weather.ds.max() <= T0
    assert view.weather_forecast.loc[T0 + timedelta(hours=1), "temperature_2m"] == 8.0
    assert view.ledger.observed_until.max() < T0
    assert view.baselines["served_forecast.parquet"].expected_w.tolist() == [1000.0]
    assert len(view.outcomes["mode_events.parquet"]) == 2
    assert view.spans("s0", device_id="hp", horizon_end=T0 + timedelta(hours=2)) == ()
    oracle = view.spans("s1", device_id="hp", horizon_end=T0 + timedelta(hours=2))
    assert oracle[0].start == T0 + timedelta(minutes=1) and oracle[0].mode == "OFF"
    assert (
        view.spans("s2", device_id="hp", horizon_end=T0 + timedelta(hours=2))[0].mode
        == "ECO"
    )
    assert (
        view.schedule_payload["generated_at"] == (T0 - timedelta(minutes=1)).isoformat()
    )
    later = bundle.available_at(T0 + timedelta(minutes=6), include_inputs=True)
    assert (
        later.schedule_payload["generated_at"]
        == (T0 + timedelta(minutes=4)).isoformat()
    )
    assert (
        later.schedule_payload["valid_until"]
        == (T0 + timedelta(minutes=19)).isoformat()
    )


def test_future_feature_perturbation_leaves_the_origin_snapshot_unchanged(bundle):
    original = bundle.available_at(T0, include_inputs=True)
    tables = {name: frame.copy(deep=True) for name, frame in bundle.tables.items()}
    tables["minutes.parquet"].loc[T0 + timedelta(minutes=1), "power_w"] = 1e8
    tables["mode_events.parquet"].loc[1, "mode"] = "BOOST"
    tables["weather_vintages.parquet"].loc[3, "temperature_2m"] = -100.0
    changed = replace(bundle, tables=tables).available_at(T0, include_inputs=True)
    pd.testing.assert_frame_equal(original.training.minutes, changed.training.minutes)
    pd.testing.assert_frame_equal(
        original.training.mode_events, changed.training.mode_events
    )
    pd.testing.assert_frame_equal(original.weather_forecast, changed.weather_forecast)
    assert original.training.lineage == changed.training.lineage


def test_previous_runs_selects_required_lead_and_day0_is_explicitly_optimistic(bundle):
    frame = bundle.tables["weather_vintages.parquet"].iloc[:2].copy()
    stale = frame.iloc[[1]].copy()
    stale["available_at"] -= timedelta(days=1)
    stale["temperature_2m"] = 99.0
    chosen = _origin_weather(pd.concat([frame, stale]), T0)
    assert chosen.loc[T0 + timedelta(hours=1), "temperature_2m"] == 7.0
    day0 = frame.iloc[[1]].copy()
    day0["vintage_kind"] = "approx_day0"
    day0["available_at"] = day0.valid_at
    chosen = _origin_weather(day0, T0)
    assert chosen.iloc[0].available_at == T0
    assert chosen.iloc[0].original_available_at == T0 + timedelta(hours=1)
    assert chosen.iloc[0].vintage_kind == "approx_day0"
    day0["vintage_kind"] = "unknown"
    with pytest.raises(ValueError, match="unknown weather"):
        _origin_weather(day0, T0)
