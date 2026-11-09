"""Lifecycle, family fallbacks and origin-availability contract regressions."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.cycle_model import CycleFit, CycleParameters
from kpower_forecast.hvac.model import (
    KPowerHvacForecast,
    _predict_candidate,
    _State,
    fingerprint,
)
from kpower_forecast.hvac.power import PowerFit
from kpower_forecast.hvac.types import (
    FitResult,
    GridInterval,
    HistoryPolicy,
    HvacCancelled,
    HvacCapabilities,
    HvacFamily,
    HvacForecastRequest,
    HvacModelConfig,
    HvacTrainingData,
    InputTier,
    MeteredScope,
)


@pytest.fixture
def training():
    index = pd.date_range("2026-11-01", periods=3 * 1440, freq="min", tz="UTC")
    wave = 2 * np.pi * np.arange(len(index)) / 1440
    temperature = 7 + 3 * np.sin(wave)
    minutes = pd.DataFrame(
        {
            "power_w": 1300 + 60 * (10 - temperature),
            "outdoor_env_c": temperature,
            "power_w__measured_valid": True,
            "power_w__quality": "good",
            "outdoor_env_c__measured_valid": True,
            "outdoor_env_c__quality": "good",
        },
        index=index,
    )
    hours = pd.date_range(index[0], periods=100, freq="h")
    weather = pd.DataFrame(
        {
            "temperature_2m": 7 + 3 * np.sin(2 * np.pi * np.arange(100) / 24),
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 2.0,
            "weather_source": "historical_forecast",
        },
        index=hours,
    )
    reference = json.loads(
        (Path(__file__).parent / "fixtures/hvac_owner_policy.json").read_text()
    )
    policy = HistoryPolicy(
        version=reference["version"],
        scope="direct_device",
        settings=reference["settings"],
        measured_valid_qualities=reference["measured_valid_qualities"],
    )
    data = HvacTrainingData(
        HvacCapabilities(
            family="inverter", fingerprint="synthetic", metered_scope="whole_unit"
        ),
        minutes,
        pd.DataFrame({"time": [index[0]], "mode": ["NORMAL"], "source": ["manual"]}),
        weather,
        index[0].to_pydatetime(),
        {"sensor": "env-a"},
        policy,
    )
    cfg = HvacModelConfig(
        model_id="fixture",
        timezone="UTC",
        n_paths=32,
        n_parameter_sets=1,
        ingest_lag_min=2,
    )
    origin = (index[-1] + timedelta(minutes=3)).to_pydatetime()
    grid = tuple(
        GridInterval(start=origin + timedelta(minutes=i), minutes=5)
        for i in range(0, 120, 5)
    )
    request = HvacForecastRequest(origin, minutes, (), "NORMAL", weather, grid)
    return data, cfg, origin, request


def test_fit_install_save_load_are_separate_and_reject_incompatible_artifacts(
    training, tmp_path
):
    data, cfg, origin, request = training
    directory = tmp_path / "not-created"
    model = KPowerHvacForecast(cfg, storage_dir=directory)
    candidate = model.fit(data, as_of=origin)
    assert model.active_fit is None and not directory.exists()
    assert candidate.diagnostics["reliable"] is False
    assert candidate.diagnostics["reason"] == "insufficient_history"
    with pytest.raises(ValueError, match="no installed"):
        model.predict(request)
    model.install(candidate)
    assert not directory.exists()
    forecast = model.predict(request)
    assert forecast.tier is InputTier.T0_ENERGY
    assert forecast.fallback_reason == "family_not_implemented:inverter"
    assert forecast.intervals.run_probability is None
    assert forecast.paths_w.shape == (0, len(request.grid))
    model.save()
    artifact = model.artifact_path
    document = json.loads(artifact.read_text())
    assert "minute_paths" not in document and "paths_w" not in document
    restored = KPowerHvacForecast(cfg, storage_dir=directory)
    assert restored.load(expected_fingerprint=candidate.fingerprint)
    np.testing.assert_array_equal(
        forecast.minute_paths.power_w, restored.predict(request).minute_paths.power_w
    )
    assert not restored.load(expected_fingerprint="changed")
    assert restored.active_fit.fingerprint == candidate.fingerprint
    snapshot = restored.active_fit
    snapshot.parameters["lineage"]["sensor"] = "mutated"
    assert restored.active_fit.parameters["lineage"]["sensor"] == "env-a"
    document["parameters"]["energy"]["coefficients"] = []
    artifact.write_text(json.dumps(document))
    assert not restored.load(expected_fingerprint=candidate.fingerprint)
    assert restored.active_fit.fingerprint == candidate.fingerprint
    artifact.write_text("{broken")
    assert not restored.load(expected_fingerprint=candidate.fingerprint)
    assert not KPowerHvacForecast(cfg, storage_dir=tmp_path / "absent").load(
        expected_fingerprint=candidate.fingerprint
    )
    assert (
        fingerprint(replace(data, lineage={"sensor": "new"}), cfg)
        != candidate.fingerprint
    )
    changed_policy = data.history_policy.model_copy(update={"version": 4})
    assert (
        fingerprint(replace(data, history_policy=changed_policy), cfg)
        != candidate.fingerprint
    )
    assert (
        fingerprint(data, cfg.model_copy(update={"timezone": "Europe/Ljubljana"}))
        != candidate.fingerprint
    )


@pytest.mark.parametrize("family", ["two_stage", "multi_stage", "inverter", "unknown"])
def test_family_compatibility_never_invents_compressor_probabilities(training, family):
    data, cfg, origin, request = training
    data = replace(
        data,
        capabilities=data.capabilities.model_copy(
            update={"family": HvacFamily(family)}
        ),
    )
    model = KPowerHvacForecast(cfg)
    candidate = model.fit(data, as_of=origin)
    model.install(candidate)
    prediction = model.predict(request)
    assert prediction.tier is InputTier.T0_ENERGY
    assert prediction.intervals.run_probability is None
    assert prediction.intervals.start_probability is None
    assert prediction.minute_paths.running is None
    assert prediction.fallback_reason == (
        "not_cycling" if family == "unknown" else f"family_not_implemented:{family}"
    )


def test_future_values_events_and_weather_cannot_affect_origin_forecast(training):
    data, cfg, origin, request = training
    model = KPowerHvacForecast(cfg)
    baseline = model.fit(data, as_of=origin)
    model.install(baseline)
    expected = model.predict(request)
    future = data.minutes.iloc[-20:].copy()
    future.index += timedelta(days=1)
    future["power_w"] = 100000.0
    future["outdoor_env_c"] = -100.0
    altered = replace(
        data,
        minutes=pd.concat([data.minutes, future]),
        mode_events=pd.concat(
            [
                data.mode_events,
                pd.DataFrame(
                    {
                        "time": [origin + timedelta(minutes=1)],
                        "mode": ["OFF"],
                        "source": ["manual"],
                    }
                ),
            ]
        ),
    )
    after = model.fit(altered, as_of=origin)
    assert after.parameters == baseline.parameters
    # Availability, rather than valid time alone, excludes a delayed forecast.
    weather = request.weather_forecast.copy()
    weather["available_at"] = origin - timedelta(minutes=1)
    spike = weather.copy()
    spike["temperature_2m"] = -100.0
    spike["available_at"] = origin + timedelta(minutes=1)
    prediction = model.predict(
        replace(
            request,
            recent=altered.minutes,
            weather_forecast=pd.concat([weather, spike]),
        )
    )
    np.testing.assert_array_equal(
        expected.minute_paths.power_w, prediction.minute_paths.power_w
    )
    cancel = Event()
    cancel.set()
    with pytest.raises(HvacCancelled):
        model.fit(data, as_of=origin, cancel=cancel)
    with pytest.raises(HvacCancelled):
        model.predict(request, cancel=cancel)
    assert model.active_fit.fingerprint == baseline.fingerprint


def test_fixed_speed_candidate_predicts_all_paths_and_recomputes_statistics(training):
    data, cfg, origin, request = training
    model = KPowerHvacForecast(cfg)
    energy_candidate = model.fit(data, as_of=origin)
    data = replace(
        data,
        capabilities=data.capabilities.model_copy(
            update={"family": HvacFamily.FIXED_SPEED}
        ),
    )
    state = _State.model_validate(dict(energy_candidate.parameters))
    p = CycleParameters(
        20.0,
        (),
        (np.log(45 * 13), 0.0, 0.0, 0.0),
        0.4,
        1 / 45 + 1 / 15,
        0.0,
        (),
        (),
        0.2,
    )
    state = state.model_copy(
        update={
            "capabilities": data.capabilities,
            "energy": None,
            "cycle": CycleFit((p,), (), False, 0, ()),
            "threshold_w": 200.0,
            "power": PowerFit(
                (4000.0, 0.0),
                False,
                0.0,
                (1.0, 1.0, 1.0),
                {},
                100.0,
                50.0,
                None,
                MeteredScope.WHOLE_UNIT,
                (),
            ),
        }
    )
    candidate = FitResult(
        origin,
        fingerprint(data, cfg),
        InputTier.T1_POWER,
        HvacFamily.FIXED_SPEED,
        state.model_dump(mode="json"),
        {},
        None,
    )
    model.install(candidate)
    result = model.predict(request)
    assert result.minute_paths.power_w.shape == (32, 120)
    assert result.diagnostics["n_paths"] == 32
    assert result.diagnostics["run_threshold_w"] == 200.0
    unknown = _predict_candidate(request, candidate, cfg, force_phase_unknown=True)
    assert unknown.diagnostics["phase_forced_unknown"] is True
    assert unknown.diagnostics["stationary_initial_state"] is True
    assert "phase_forced_unknown" not in result.diagnostics
    assert result.paths_w.shape == (16, 24)
    assert result.intervals.run_probability is not None
    quarter = tuple(
        GridInterval(start=origin + timedelta(minutes=i), minutes=15)
        for i in range(0, 120, 15)
    )
    assert result.stats(quarter).expected_kwh.sum() == pytest.approx(
        result.intervals.expected_kwh.sum()
    )
    with pytest.raises(ValueError, match="precedes"):
        model.predict(replace(request, origin=origin - timedelta(minutes=1)))


def test_fixed_speed_fit_to_prediction_uses_the_physical_model(training):
    data, cfg, origin, request = training
    minutes = data.minutes.copy()
    rng = np.random.default_rng(29)
    power = np.empty(len(minutes))
    position = 0
    running = False
    while position < len(minutes):
        temperature = minutes.outdoor_env_c.iloc[position]
        a = (20 - temperature) / (45 * 13)
        median = 1 / (1 / 45 + 1 / 15 - a) if running else 1 / a
        length = max(1, int(round(median * np.exp(0.3 * rng.normal()))))
        power[position : position + length] = 4000.0 if running else 100.0
        position += length
        running = not running
    minutes["power_w"] = power
    cap = HvacCapabilities(
        family="fixed_speed",
        fingerprint="rated",
        metered_scope="whole_unit",
        rated_points=(
            {
                "source_temp_c": 0.0,
                "supply_temp_c": 35.0,
                "heat_output_kw": 10.0,
                "electrical_input_kw": 3.0,
            },
            {
                "source_temp_c": 7.0,
                "supply_temp_c": 35.0,
                "heat_output_kw": 10.0,
                "electrical_input_kw": 3.0,
            },
        ),
    )
    data = replace(data, minutes=minutes, capabilities=cap)
    model = KPowerHvacForecast(cfg)
    candidate = model.fit(data, as_of=origin)
    assert candidate.family is HvacFamily.FIXED_SPEED
    assert candidate.tier is InputTier.T1_POWER and candidate.fallback_reason is None
    assert candidate.parameters["cycle"] is not None
    model.install(candidate)
    forecast = model.predict(replace(request, recent=minutes))
    assert forecast.minute_paths.running is not None
    assert forecast.minute_paths.running.any()
    assert (
        forecast.diagnostics["reliable"] is False
    )  # 3d never waives the owner 14d policy


def test_t3_fit_learns_mode_conditioned_thresholds_from_command_history(training):
    data, cfg, _, _ = training
    index = pd.date_range("2026-10-01", periods=15 * 1440, freq="min", tz="UTC")
    temperature = 7 + 3 * np.sin(2 * np.pi * np.arange(len(index)) / 1440)
    eco = (np.arange(len(index)) >= 5 * 1440) & (np.arange(len(index)) < 10 * 1440)
    setpoint = 40 + 0.5 * temperature + 5 * (~eco)
    rng = np.random.default_rng(67)
    power = np.empty(len(index))
    position = 0
    running = False
    while position < len(index):
        length = max(
            3, int(round((20 if running else 50) * np.exp(0.2 * rng.normal())))
        )
        power[position : position + length] = 4600.0 if running else 100.0
        position += length
        running = not running
    minutes = pd.DataFrame(
        {
            "power_w": power,
            "outdoor_env_c": temperature,
            "control_c": 42 + np.sin(2 * np.pi * np.arange(len(index)) / 90),
            "control_setpoint_c": setpoint,
            "control_hysteresis_c": 3.0,
        },
        index=index,
    )
    for name in list(minutes.columns):
        minutes[name + "__measured_valid"] = True
        minutes[name + "__quality"] = "good"
    events = pd.DataFrame(
        {
            "time": [index[0], index[5 * 1440], index[10 * 1440]],
            "mode": ["NORMAL", "ECO", "NORMAL"],
            "source": "manual",
        }
    )
    hours = pd.date_range(index[0], periods=15 * 24 + 2, freq="h")
    weather = pd.DataFrame(
        {
            "temperature_2m": 7 + 3 * np.sin(2 * np.pi * np.arange(len(hours)) / 24),
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 2.0,
            "weather_source": "historical_forecast",
        },
        index=hours,
    )
    cap = data.capabilities.model_copy(
        update={"family": HvacFamily.FIXED_SPEED, "run_threshold_w": 2000.0}
    )
    data = replace(
        data,
        minutes=minutes,
        mode_events=events,
        weather=weather,
        capabilities=cap,
        epoch_start=index[0].to_pydatetime(),
    )
    cfg = cfg.model_copy(update={"enable_t3": True})
    candidate = KPowerHvacForecast(cfg).fit(
        data, as_of=(index[-1] + timedelta(minutes=3)).to_pydatetime()
    )
    assert candidate.tier is InputTier.T3_CONTROLLER
    controller = candidate.parameters["controller"]
    assert controller["threshold_modes"] == ["ECO", "NORMAL"]
    np.testing.assert_allclose(
        controller["threshold_beta"], [40.0, 0.5, 5.0], atol=1e-6
    )
    assert candidate.parameters["power"]["temperature_source"] == "control_setpoint_c"
