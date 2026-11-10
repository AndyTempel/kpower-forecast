"""Rolling scoring uses one causal gateway and never trains on future inputs."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac._artifacts import write_report
from kpower_forecast.hvac._backtest import backtest
from kpower_forecast.hvac._report import summarize
from kpower_forecast.hvac.bundle import BundleManifest, HvacEvaluationBundle
from kpower_forecast.hvac.model import KPowerHvacForecast
from kpower_forecast.hvac.types import HvacModelConfig

ORIGIN = datetime(2026, 11, 4, tzinfo=UTC)


@pytest.fixture
def bundle():
    owner = json.loads(
        (Path(__file__).parent / "fixtures/hvac_owner_policy.json").read_text()
    )
    index = pd.date_range(ORIGIN - timedelta(days=3), periods=3 * 1440 + 60, freq="min")
    temp = 7 + 3 * np.sin(np.arange(len(index)) * 2 * np.pi / 1440)
    minutes = pd.DataFrame(
        {
            "power_w": 1800 - 60 * temp,
            "power_w__quality": "good",
            "power_w__measured_valid": True,
            "outdoor_env_c": temp,
            "outdoor_env_c__quality": "good",
            "outdoor_env_c__measured_valid": True,
        },
        index=index,
    )
    hours = pd.date_range(index[0], periods=3 * 24 + 130, freq="h")
    weather = pd.DataFrame(
        {
            "ds": hours,
            "temperature_2m": 7 + 3 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24),
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 2.0,
            "weather_source": "historical_forecast",
        }
    )
    vintage = weather.copy()
    vintage["valid_at"] = vintage.ds
    vintage["available_at"] = ORIGIN - timedelta(hours=1)
    vintage["vintage_kind"] = "archived"
    manifest = BundleManifest(
        exported_at=ORIGIN + timedelta(days=1),
        start=index[0],
        end=index[-1] + timedelta(minutes=1),
        controller_id="fixture",
        device_id="hp",
        timezone="UTC",
        latitude=0,
        longitude=0,
        ems_commit="1234567",
        ingest_lag_min=2,
        training_ready=True,
        files={},
    )
    return HvacEvaluationBundle(
        manifest,
        {
            "minutes.parquet": minutes,
            "weather.parquet": weather,
            "weather_vintages.parquet": vintage,
        },
        {
            "history_policy.json": {
                "minute_history_policy": owner["settings"],
                "measured_valid_qualities": owner["measured_valid_qualities"],
            },
            "minute_qualification.json": {"scope": "direct_device_minutes"},
            "capabilities.json": {"family": "inverter", "fingerprint": "fixture"},
        },
    )


def test_future_only_feature_spike_cannot_change_full_scoring_and_gaps_cannot_pass_gate(
    bundle,
    tmp_path,
):
    cfg = HvacModelConfig(
        model_id="test",
        timezone="UTC",
        ingest_lag_min=2,
        n_paths=16,
    )
    original = backtest(bundle, cfg, [ORIGIN], phase_ablation=False)
    changed = bundle.tables["minutes.parquet"].copy()
    changed.loc[changed.index >= ORIGIN, "outdoor_env_c"] = 10000
    mutated = replace(bundle, tables={**bundle.tables, "minutes.parquet": changed})
    perturbed = backtest(mutated, cfg, [ORIGIN], phase_ablation=False)
    for name in ("quarters", "runs", "energy", "cadence", "blocks"):
        pd.testing.assert_frame_equal(original.tables[name], perturbed.tables[name])
    report = summarize(original.tables, n_paths=16)
    assert report["gates"]["A4"]["status"] == "unscorable"
    assert report["gates"]["A7"]["status"] == "unscorable"
    assert (
        original.tables["runs"]
        .loc[lambda rows: rows.model.eq("cycle")]
        .probability.isna()
        .all()
    )
    output = tmp_path / "report"
    write_report(original, output)
    assert json.loads((output / "metrics.json").read_text())["scope"] == "development"
    assert "cannot satisfy A4" in (output / "report.md").read_text()
    assert not any(
        "path" in column for column in pd.read_parquet(output / "quarters.parquet")
    )
    with pytest.raises(FileExistsError):
        write_report(original, output)


def test_one_gateway_per_origin_and_six_hour_or_lineage_refit(bundle, monkeypatch):
    cfg = HvacModelConfig(
        model_id="test",
        timezone="UTC",
        ingest_lag_min=2,
        n_paths=16,
    )
    gateway_calls, fit_calls = [], []
    original_gateway = HvacEvaluationBundle.available_at
    original_fit = KPowerHvacForecast.fit

    def gateway(self, origin, *, include_inputs=False):
        gateway_calls.append(origin)
        assert include_inputs is True
        view = original_gateway(self, origin, include_inputs=True)
        if origin >= ORIGIN + timedelta(hours=7):
            return replace(
                view, training=replace(view.training, lineage={"sensor": "changed"})
            )
        return view

    def fit(self, data, *, as_of, **kwargs):
        fit_calls.append(as_of)
        return original_fit(self, data, as_of=as_of, **kwargs)

    monkeypatch.setattr(HvacEvaluationBundle, "available_at", gateway)
    monkeypatch.setattr(KPowerHvacForecast, "fit", fit)
    origins = [
        ORIGIN,
        ORIGIN + timedelta(minutes=15),
        ORIGIN + timedelta(hours=6),
        ORIGIN + timedelta(hours=7),
    ]
    result = backtest(bundle, cfg, origins, phase_ablation=False)
    assert gateway_calls == origins
    assert fit_calls == [origins[0], origins[2], origins[3]]
    assert len(result.tables["fits"]) == 3
    skipped = backtest(bundle, cfg, [ORIGIN], schedule="s2")
    assert skipped.tables["quarters"].empty
    assert skipped.tables["origins"].status.tolist() == ["skipped_missing_schedule"]
