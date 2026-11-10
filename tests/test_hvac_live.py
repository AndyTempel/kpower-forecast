"""Live acceptance uses original published numbers, never a model refit."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import test_hvac_backtest as shared

from kpower_forecast.hvac._live import live_score
from kpower_forecast.hvac._report import summarize
from kpower_forecast.hvac.model import KPowerHvacForecast
from kpower_forecast.hvac.types import HvacModelConfig

ORIGIN = shared.ORIGIN
source_bundle = shared.bundle


@pytest.fixture
def published(source_bundle):
    bundle = source_bundle
    origin = ORIGIN - timedelta(minutes=1)
    stamps = pd.date_range(ORIGIN, periods=4, freq="15min")
    forecasts = pd.DataFrame(
        {
            "origin": origin,
            "interval_start": stamps,
            "horizon": "hourly",
            "expected_w": 1000.0,
            "q10_w": 400.0,
            "q50_w": 800.0,
            "q90_w": 2000.0,
            "run_probability": 0.6,
            "n_paths": 256,
            "run_threshold_w": 200.0,
        }
    )
    near = pd.DataFrame(
        {
            "origin": origin,
            "interval_start": pd.date_range(ORIGIN, periods=24, freq="5min"),
            "run_probability": 0.6,
        }
    )
    site_index = bundle.tables["minutes.parquet"].index
    site = pd.DataFrame(
        {"base_w": 0.0, "pv_w": 0.0, "base_coverage_s": 60.0}, index=site_index
    )
    reserve = pd.DataFrame(
        [
            {
                "generated_at": ORIGIN + timedelta(seconds=1),
                "forecast_origin": origin,
                "plan": plan,
                "block_id": "a",
                "window_start": ORIGIN,
                "window_end": ORIGIN + timedelta(minutes=15),
                "quantile": 0.9,
                "energy_wh": 123.0,
                "hard_energy_wh": 111.0,
            }
            for plan in ("strategic", "near_term")
        ]
    )
    inputs = pd.DataFrame(
        [
            {
                "generated_at": ORIGIN + timedelta(seconds=1),
                "forecast_origin": origin,
                "plan": plan,
                "block_id": "a",
                "start": ORIGIN,
                "end": ORIGIN + timedelta(minutes=15),
                "import_allowance_wh": -25.0,
                "hard_import_limit_w": 0.0,
            }
            for plan in ("strategic", "near_term")
        ]
    )
    return replace(
        bundle,
        tables={
            **bundle.tables,
            "live_forecasts.parquet": forecasts,
            "live_first_hour.parquet": near,
            "live_reserves.parquet": reserve,
            "live_window_inputs.parquet": inputs,
            "site_minutes.parquet": site,
        },
    )


def test_live_scores_published_quantiles_reserves_and_unknown_original_evidence(
    published, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("live scoring must not fit or predict")

    monkeypatch.setattr(KPowerHvacForecast, "fit", forbidden)
    monkeypatch.setattr(KPowerHvacForecast, "predict", forbidden)
    cfg = HvacModelConfig(model_id="test", timezone="UTC", ingest_lag_min=2)
    result = live_score(published, cfg, deploy=ORIGIN - timedelta(minutes=1))
    cycle = result.tables["quarters"].loc[
        lambda rows: rows.model.eq("cycle") & rows.eligible
    ]
    assert len(cycle) == 4
    assert cycle.q50_kwh.tolist() == pytest.approx([0.2] * 4)
    blocks = result.tables["blocks"].loc[lambda rows: rows.model.eq("cycle")]
    assert blocks.eligible.all() and blocks.reserve_q90_wh.tolist() == [123.0, 123.0]
    actual = (
        published.tables["minutes.parquet"]
        .loc[
            lambda rows: (
                (rows.index >= ORIGIN) & (rows.index < ORIGIN + timedelta(minutes=15))
            ),
            "power_w",
        ]
        .sum()
        / 60
    )
    assert blocks.actual_need_wh.tolist() == pytest.approx([actual, actual])
    report = summarize(
        result.tables, n_paths=999
    )  # never substitutes for published size
    assert report["gates"]["A4"]["status"] == "unscorable"
    assert result.tables["energy"].loc[
        lambda rows: rows.model.eq("cycle") & rows.energy_scope.eq("lead_diagnostic")
    ].partial_method.unique().tolist() == ["published_quarter_mean_piecewise_constant"]
    official = result.tables["energy"].loc[
        lambda rows: rows.model.eq("cycle") & rows.energy_scope.eq("first_horizon")
    ]
    assert len(official) == 3 and not official.eligible.any()
    changed = published.tables["live_forecasts.parquet"].copy()
    changed[["n_paths", "run_threshold_w"]] = np.nan
    missing = replace(
        published,
        tables={
            **published.tables,
            "live_forecasts.parquet": changed,
            "live_window_inputs.parquet": pd.DataFrame(),
        },
    )
    result = live_score(missing, cfg, deploy=ORIGIN - timedelta(minutes=1))
    assert not result.tables["runs"].eligible.any()
    assert not result.tables["blocks"].eligible.any()
    assert (
        summarize(result.tables, n_paths=999)["gates"]["A5"]["status"] == "unscorable"
    )


def test_live_rejects_duplicate_or_conflicting_published_identity(published):
    cfg = HvacModelConfig(model_id="test", timezone="UTC", ingest_lag_min=2)
    source = published.tables["live_forecasts.parquet"]
    duplicate = replace(
        published,
        tables={
            **published.tables,
            "live_forecasts.parquet": pd.concat(
                [source, source.iloc[:1]], ignore_index=True
            ),
        },
    )
    with pytest.raises(ValueError, match="duplicate published"):
        live_score(duplicate, cfg, deploy=ORIGIN - timedelta(minutes=1))


def test_original_horizon_total_scores_missing_grid_prefix_without_model_or_gap_fill(
    published, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("must use published horizon total")

    monkeypatch.setattr(KPowerHvacForecast, "predict", forbidden)
    cfg = HvacModelConfig(model_id="test", timezone="UTC", ingest_lag_min=2)
    origin = ORIGIN - timedelta(minutes=1)
    totals = pd.DataFrame(
        [{"origin": origin, "horizon": "hourly", "duration_h": 1, "expected_kwh": 0.9}]
    )
    source = replace(
        published, tables={**published.tables, "live_horizon_energy.parquet": totals}
    )
    result = live_score(source, cfg, deploy=origin)
    row = (
        result.tables["energy"]
        .loc[
            lambda rows: (
                rows.model.eq("cycle")
                & rows.energy_scope.eq("first_horizon")
                & rows.duration_h.eq(1)
            )
        ]
        .iloc[0]
    )
    assert row.interval_start == origin and row.eligible
    assert row.expected_kwh == 0.9 and row.horizon_evidence == "original_mean_total"
    actual = published.tables["minutes.parquet"].copy()
    actual.loc[origin, "power_w"] = np.nan
    gapped = replace(source, tables={**source.tables, "minutes.parquet": actual})
    result = live_score(gapped, cfg, deploy=origin)
    row = (
        result.tables["energy"]
        .loc[
            lambda rows: (
                rows.model.eq("cycle")
                & rows.energy_scope.eq("first_horizon")
                & rows.duration_h.eq(1)
            )
        ]
        .iloc[0]
    )
    assert not row.eligible and row.coverage == pytest.approx(59 / 60)
    assert np.isnan(row.error_kwh) and np.isnan(row.partial_error_kwh)
