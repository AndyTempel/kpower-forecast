"""Deterministic checks for sparse-observation thermal identification."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kpower_forecast.thermal import (
    KPowerThermalForecast,
    ThermalModelConfig,
    ThermalTrainingTransition,
)


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
    uncovered = rows[2].model_copy(update={"hvac_coverage_ratio": 0.8})
    d = _model(tmp_path).train([long_row, short_row, uncovered, *rows[3:]])
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
    assert not d.fitted
    assert d.unreliable_reason == "implausible_parameters"


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
