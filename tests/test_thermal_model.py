"""Deterministic checks for sparse-observation thermal identification."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kpower_forecast.thermal import (
    NAIVE_SOURCE,
    KPowerThermalForecast,
    ThermalModelConfig,
    ThermalTrainingTransition,
    predict_naive,
    recent_trend_c_per_hour,
)
from kpower_forecast.thermal import model as thermal_model
from kpower_forecast.weather_client import WeatherClient, WeatherConfig


def _synthetic_transitions(
    *, tau_hours: float = 22.0, gain_c_per_kw: float = 4.0
) -> list[ThermalTrainingTransition]:
    """Generate real endpoint pairs with irregular spacing and on/off drive."""
    rng = np.random.default_rng(73)
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    indoor = 19.0
    rows = []
    for index in range(190):
        duration = (0.5, 0.75, 1.0, 1.25)[index % 4]
        outside = 1.0 + 7 * np.sin(index / 13)
        power = 1900.0 if index % 11 < 6 else 0.0
        decay = np.exp(-duration / tau_hours)
        next_indoor = (
            decay * indoor
            + (1 - decay) * (outside + gain_c_per_kw * power / 1000 + 14.0)
            + rng.normal(0, 0.015)
        )
        end = instant + timedelta(hours=duration)
        rows.append(
            ThermalTrainingTransition(
                start_at=instant,
                end_at=end,
                indoor_start_c=indoor,
                indoor_end_c=next_indoor,
                outdoor_mean_c=outside,
                hvac_electric_mean_w=power,
                hvac_coverage_ratio=1.0,
            )
        )
        instant, indoor = end, next_indoor
    return rows


def _model(tmp_path: object) -> KPowerThermalForecast:
    """Construct a model with bounded synthetic-test parameters."""
    return KPowerThermalForecast(
        model_id="thermal_aggregate",
        authority_fingerprint="epoch-1",
        storage_path=tmp_path,
        config=ThermalModelConfig(max_equilibrium_offset_c=16),
    )


def test_irregular_fit_prediction_and_restore(tmp_path: object) -> None:
    """Sparse irregular targets recover stable dynamics without a target grid."""
    model = _model(tmp_path)
    diagnostics = model.train(_synthetic_transitions())
    assert diagnostics.fitted
    assert diagnostics.reliable_on_holdout
    assert diagnostics.time_constant_hours == pytest.approx(22, rel=0.3)
    assert diagnostics.effective_gain_c_per_kw == pytest.approx(4, rel=0.3)
    assert diagnostics.hvac_on_transitions > 0
    assert diagnostics.hvac_off_transitions > 0
    assert diagnostics.holdout_horizon_metrics["1h"]["count"] > 0
    assert diagnostics.holdout_horizon_metrics["3h"]["count"] > 0
    assert diagnostics.holdout_horizon_metrics["6h"]["count"] > 0
    assert diagnostics.holdout_horizon_metrics["12h"]["count"] > 0
    origin = datetime(2026, 2, 1, tzinfo=timezone.utc)
    prediction = model.predict(
        origin=origin,
        initial_temperature_c=18,
        outdoor_temperature_c=[0] * 12,
        hvac_electric_power_w=[2000] * 12,
        outdoor_source="archive-model",
        hvac_drive_source="legacy-heating",
    )
    assert prediction.intervals[0].timestamp == origin + timedelta(minutes=15)
    assert prediction.intervals[-1].timestamp == origin + timedelta(hours=3)
    assert prediction.intervals[0].indoor_temperature_c > 18
    assert (
        prediction.intervals[-1].upper_temperature_c
        - prediction.intervals[-1].lower_temperature_c
        > prediction.intervals[0].upper_temperature_c
        - prediction.intervals[0].lower_temperature_c
    )
    model.save()
    restored = _model(tmp_path)
    assert restored.load()
    assert (
        restored.predict(
            origin=origin,
            initial_temperature_c=18,
            outdoor_temperature_c=[0] * 12,
            hvac_electric_power_w=[2000] * 12,
            outdoor_source="archive-model",
            hvac_drive_source="legacy-heating",
        )
        == prediction
    )
    incompatible = KPowerThermalForecast(
        model_id="thermal_aggregate",
        authority_fingerprint="epoch-2",
        storage_path=tmp_path,
        config=model.config,
    )
    assert not incompatible.load()


def test_coverage_and_duration_gaps_reject_transitions(tmp_path: object) -> None:
    """Long, short and uncovered spans never become training observations."""
    rows = _synthetic_transitions()
    long_row = rows[0].model_copy(
        update={"end_at": rows[0].start_at + timedelta(hours=7)}
    )
    short_row = rows[1].model_copy(
        update={"end_at": rows[1].start_at + timedelta(minutes=2)}
    )
    uncovered = rows[2].model_copy(update={"hvac_coverage_ratio": 0.89})
    # A meter timeout costs a few seconds per bucket; such rows stay usable.
    timeout = rows[3].model_copy(update={"hvac_coverage_ratio": 0.9})
    d = _model(tmp_path).train([long_row, short_row, uncovered, timeout, *rows[4:]])
    assert d.rejected_long == 1
    assert d.rejected_short == 1
    assert d.rejected_hvac_coverage == 1
    assert d.transition_count == len(rows) - 3


def test_unexcited_and_implausible_fit_fail_closed(tmp_path: object) -> None:
    """Flat targets and negative heating response cannot earn reliability."""
    rows = _synthetic_transitions()
    flat = [
        row.model_copy(update={"indoor_start_c": 20.0, "indoor_end_c": 20.0})
        for row in rows
    ]
    d = _model(tmp_path).train(flat)
    assert not d.fitted
    assert d.unreliable_reason == "insufficient_temperature_variation"
    negative_response = _synthetic_transitions(gain_c_per_kw=-4.0)
    d = _model(tmp_path).train(negative_response)
    # Outdoor forcing still makes the bounded fit useful, but the heating
    # response is unidentified: published as low quality, never accepted.
    assert d.fitted
    assert not d.reliable_on_holdout
    assert d.quality == "low_identifiability"
    assert d.unreliable_reason == "low_identifiability"
    assert "gain_constrained" in d.identifiability_flags
    assert d.effective_gain_c_per_kw == pytest.approx(
        ThermalModelConfig().min_effective_gain_c_per_kw
    )
    assert d.transition_fit is not None
    assert d.transition_fit["effective_gain_c_per_kw"] < 0


def test_failed_retrain_keeps_live_and_saved_fit(tmp_path: object) -> None:
    """A failed scheduled retry cannot change prediction or saved fit."""
    model = _model(tmp_path)
    assert model.train(_synthetic_transitions()).fitted
    model.save()
    origin = datetime(2026, 2, 1, tzinfo=timezone.utc)
    inputs = {
        "origin": origin,
        "initial_temperature_c": 18.0,
        "outdoor_temperature_c": [0.0] * 4,
        "hvac_electric_power_w": [2000.0] * 4,
        "outdoor_source": "weather",
        "hvac_drive_source": "heating",
    }
    previous_diagnostics = model.diagnostics.model_copy(deep=True)
    previous_prediction = model.predict(**inputs)

    failed = model.train([])
    assert not failed.fitted
    assert failed.unreliable_reason == "insufficient_history"
    assert model.diagnostics == previous_diagnostics
    assert model.predict(**inputs) == previous_prediction

    model.save()
    restored = _model(tmp_path)
    assert restored.load()
    assert restored.diagnostics == previous_diagnostics
    assert restored.predict(**inputs) == previous_prediction


def test_prediction_rejects_missing_or_invalid_drive(tmp_path: object) -> None:
    """A trained model cannot infer or extend an absent future HVAC drive."""
    model = _model(tmp_path)
    assert model.train(_synthetic_transitions()).fitted
    with pytest.raises(ValueError, match="grids differ"):
        model.predict(
            origin=datetime(2026, 2, 1, tzinfo=timezone.utc),
            initial_temperature_c=20,
            outdoor_temperature_c=[1],
            hvac_electric_power_w=[1000, 1000],
            outdoor_source="weather",
            hvac_drive_source="heating",
        )
    with pytest.raises(ValueError, match="finite"):
        model.predict(
            origin=datetime(2026, 2, 1, tzinfo=timezone.utc),
            initial_temperature_c=20,
            outdoor_temperature_c=[float("nan")],
            hvac_electric_power_w=[1000],
            outdoor_source="weather",
            hvac_drive_source="heating",
        )


def test_corrupt_artifact_is_rejected(tmp_path: object) -> None:
    """A damaged completion marker never makes a model look loadable."""
    model = _model(tmp_path)
    assert model.train(_synthetic_transitions()).fitted
    model.save()
    marker = next(tmp_path.glob("*_thermal_manifest.json"))
    marker.write_text("{broken", encoding="utf-8")
    assert not _model(tmp_path).load()


def test_injected_weather_location_is_bound_to_artifact(tmp_path: object) -> None:
    """The same-config artifact cannot be restored for a different site."""
    first_client = WeatherClient(46.0, 14.0, WeatherConfig(cache_enabled=False))
    model = KPowerThermalForecast(
        model_id="thermal_aggregate",
        authority_fingerprint="epoch-1",
        storage_path=tmp_path,
        config=ThermalModelConfig(max_equilibrium_offset_c=16),
        weather_client=first_client,
    )
    assert model.latitude == 46.0
    assert model.longitude == 14.0
    assert model.train(_synthetic_transitions()).fitted
    model.save()
    other_client = WeatherClient(47.0, 15.0, WeatherConfig(cache_enabled=False))
    incompatible = KPowerThermalForecast(
        model_id="thermal_aggregate",
        authority_fingerprint="epoch-1",
        storage_path=tmp_path,
        config=model.config,
        weather_client=other_client,
    )
    assert not incompatible.load()
    with pytest.raises(ValueError, match="differs from site"):
        KPowerThermalForecast(
            model_id="thermal_aggregate",
            authority_fingerprint="epoch-1",
            storage_path=tmp_path,
            latitude=46.0,
            longitude=14.0,
            weather_client=other_client,
        )


def _quantised(
    rows: list[ThermalTrainingTransition], step_c: float
) -> list[ThermalTrainingTransition]:
    """Round readings like a 0.1-0.2 C sensor that reports on change."""
    return [
        row.model_copy(
            update={
                "indoor_start_c": round(row.indoor_start_c / step_c) * step_c,
                "indoor_end_c": round(row.indoor_end_c / step_c) * step_c,
            }
        )
        for row in rows
    ]


def test_quantised_sensor_is_accepted_on_multi_step_skill(tmp_path: object) -> None:
    """One-step error ties persistence on coarse readings; horizons do not."""
    d = _model(tmp_path).train(_quantised(_synthetic_transitions(), 0.2))
    assert d.fitted
    assert d.quality == "accepted"
    assert d.reliable_on_holdout
    assert d.skill_horizons_passed >= 2
    for key in ("3h", "6h", "12h"):
        metrics = d.holdout_horizon_metrics[key]
        assert metrics["mae_c"] < 0.5 * metrics["persistence_mae_c"]
    assert d.time_constant_hours == pytest.approx(22, rel=0.3)


def test_short_lineage_publishes_low_identifiability_fit(tmp_path: object) -> None:
    """Below the accepted-history gate a fit is labelled, not withheld."""
    rows = _synthetic_transitions()
    hours = 0.0
    short: list[ThermalTrainingTransition] = []
    for row in rows:
        hours += row.elapsed_hours
        if hours > 40:
            break
        short.append(row)
    model = _model(tmp_path)
    d = model.train(short)
    assert d.fitted
    assert d.quality == "low_identifiability"
    assert d.unreliable_reason == "insufficient_history"
    assert not d.reliable_on_holdout
    origin = datetime(2026, 2, 1, tzinfo=timezone.utc)
    prediction = model.predict(
        origin=origin,
        initial_temperature_c=20,
        outdoor_temperature_c=[0.0],
        hvac_electric_power_w=[0.0],
        outdoor_source="weather",
        hvac_drive_source="heating",
    )
    interval = prediction.intervals[0]
    width = interval.upper_temperature_c - interval.indoor_temperature_c
    # One step ahead: the one-step residual band, widened for low quality.
    assert d.residual_p90_c is not None
    assert width == pytest.approx(
        ThermalModelConfig().low_identifiability_interval_scale * d.residual_p90_c
    )
    assert not _model(tmp_path).train(short[:10]).fitted


def test_short_coverage_gaps_are_bridged_on_inputs_only(tmp_path: object) -> None:
    """A dropped transition does not cut every window that spans it."""
    rows = _synthetic_transitions()
    # Drop every 12th row (<= 75 min); 30-minute ones are bridged, longer
    # ones end a chain. Targets on both sides stay real readings.
    gapped = [row for index, row in enumerate(rows) if index % 12 != 4]
    d = _model(tmp_path).train(gapped)
    assert d.fitted
    assert d.quality == "accepted"
    assert d.bridged_hours > 0
    unbridged = KPowerThermalForecast(
        model_id="thermal_aggregate",
        authority_fingerprint="epoch-1",
        storage_path=tmp_path,
        config=ThermalModelConfig(max_equilibrium_offset_c=16, max_bridge_minutes=0),
    ).train(gapped)
    assert unbridged.bridged_hours == 0
    assert unbridged.window_count < d.window_count


def test_artifact_from_other_package_version_loads_for_retrain(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Package upgrades keep the published fit and request a retrain."""
    model = _model(tmp_path)
    assert model.train(_synthetic_transitions()).fitted
    model.save()
    same = _model(tmp_path)
    assert same.load()
    assert not same.needs_retrain
    monkeypatch.setattr(thermal_model, "__version__", "9999.0.0")
    upgraded = _model(tmp_path)
    assert upgraded.load()
    assert upgraded.needs_retrain
    assert upgraded.diagnostics == model.diagnostics
    monkeypatch.setattr(thermal_model, "THERMAL_CONTRACT_VERSION", 999)
    assert not _model(tmp_path).load()


def test_aggregate_shift_scores_a_zone_offset(tmp_path: object) -> None:
    """Shifting the equilibrium tracks a zone that runs warmer."""
    rows = _synthetic_transitions()
    model = _model(tmp_path)
    assert model.train(rows).fitted
    warmer = [
        row.model_copy(
            update={
                "indoor_start_c": row.indoor_start_c + 1.5,
                "indoor_end_c": row.indoor_end_c + 1.5,
            }
        )
        for row in rows[-60:]
    ]
    plain = model.evaluate_horizons(warmer)
    shifted = model.evaluate_horizons(warmer, equilibrium_shift_c=1.5)
    assert shifted["12h"]["mae_c"] < 0.5 * plain["12h"]["mae_c"]
    origin = datetime(2026, 2, 1, tzinfo=timezone.utc)
    inputs = {
        "origin": origin,
        "initial_temperature_c": 20.0,
        "outdoor_temperature_c": [0.0] * 8,
        "hvac_electric_power_w": [0.0] * 8,
        "outdoor_source": "weather",
        "hvac_drive_source": "heating",
    }
    base = model.predict(**inputs).intervals[-1].indoor_temperature_c
    moved = model.predict(**inputs, equilibrium_shift_c=1.5).intervals[-1]
    assert moved.indoor_temperature_c > base


def test_naive_fallback_damps_recent_trend() -> None:
    """The model-free fallback holds the reading and fades the trend."""
    now = datetime(2026, 2, 1, 12, tzinfo=timezone.utc)
    readings = [(now - timedelta(minutes=15 * i), 20.0 - 0.05 * i) for i in range(12)]
    trend = recent_trend_c_per_hour(readings, now=now)
    assert trend == pytest.approx(0.2)
    assert recent_trend_c_per_hour(readings[:1], now=now) == 0.0
    prediction = predict_naive(
        origin=now, initial_temperature_c=20.0, trend_c_per_hour=trend, periods=96
    )
    assert prediction.outdoor_source == NAIVE_SOURCE
    values = [item.indoor_temperature_c for item in prediction.intervals]
    assert values[0] > 20.0
    assert values[-1] == pytest.approx(20.0 + 0.2 * 2.0, abs=1e-3)
    widths = [
        item.upper_temperature_c - item.lower_temperature_c
        for item in prediction.intervals
    ]
    assert widths == sorted(widths)
    with pytest.raises(ValueError, match="grid"):
        predict_naive(
            origin=now + timedelta(minutes=1),
            initial_temperature_c=20.0,
            trend_c_per_hour=0.0,
            periods=4,
        )
