"""Simulation invariants: full ensembles, dephasing, commands and regime changes."""

from datetime import UTC, datetime, timedelta
from threading import Event

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.aggregate import interval_stats, representatives
from kpower_forecast.hvac.cycle_model import CycleFit, CycleParameters
from kpower_forecast.hvac.initial_state import InitialState
from kpower_forecast.hvac.power import PowerFit
from kpower_forecast.hvac.simulate import schedule_modes, simulate
from kpower_forecast.hvac.types import (
    GridInterval,
    HvacCancelled,
    HvacCapabilities,
    HvacForecastRequest,
    HvacModelConfig,
    HvacTimeout,
    InputTier,
    MeteredScope,
    MinutePaths,
    ModeSpan,
)

ORIGIN = datetime(2026, 11, 9, tzinfo=UTC)


@pytest.fixture
def setup():
    """Constant physical regime with independent log-normal spell dispersion."""
    cfg = HvacModelConfig(model_id="test", timezone="UTC", n_paths=2048)
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
    fit = CycleFit((p,), (), False, 0, ())
    power = PowerFit(
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
    )
    grid = tuple(
        GridInterval(start=ORIGIN + timedelta(minutes=i), minutes=15)
        for i in range(0, 720, 15)
    )
    request = HvacForecastRequest(
        ORIGIN,
        pd.DataFrame({"power_w": []}, index=pd.DatetimeIndex([], tz="UTC")),
        (),
        "NORMAL",
        pd.DataFrame(),
        grid,
        30.0,
    )
    initial = InitialState(
        np.ones(cfg.n_paths, bool),
        np.zeros(cfg.n_paths),
        np.full(cfg.n_paths, 15.0),
        0.0,
        np.zeros(cfg.n_paths, bool),
        InputTier.T1_POWER,
        False,
    )
    return cfg, p, fit, power, request, initial


def test_determinism_dephasing_and_grids(setup):
    """Known RUN becomes the distribution's duty and touch, not median duty."""
    cfg, p, fit, power, request, initial = setup
    temperature = np.full((cfg.n_paths, 720), 7.0, np.float32)
    modes = np.full(temperature.shape, "NORMAL")

    def predict():
        return simulate(
            request,
            fit,
            power,
            initial,
            cfg,
            HvacCapabilities(family="fixed_speed", fingerprint="test"),
            temperature,
            np.zeros(720),
            modes,
            np.random.default_rng(17),
        ).paths

    paths = predict()
    again = predict()
    np.testing.assert_array_equal(paths.power_w, again.power_w)
    stats = interval_stats(paths, request.grid)
    assert stats.run_probability[0] == 1.0
    duty, touch = p.stationary(7.0, "NORMAL")
    assert abs(stats.run_fraction[24:].mean() - duty) < 0.03
    assert abs(stats.run_probability[24:].mean() - touch) < 0.03
    for minutes in (1, 5, 15, 60):
        grid = tuple(
            GridInterval(start=ORIGIN + timedelta(minutes=i), minutes=minutes)
            for i in range(0, 720, minutes)
        )
        coarse = interval_stats(paths, grid)
        assert coarse.expected_kwh.sum() == pytest.approx(
            paths.power_w.sum(axis=1, dtype=float).mean() / 60000, abs=1e-10
        )
    shown, display = representatives(paths, request.grid)
    assert shown.shape == (16, 48) and 0 <= display < 16
    assert not shown.flags.writeable
    cancelled = Event()
    cancelled.set()
    with pytest.raises(HvacCancelled):
        simulate(
            request,
            fit,
            power,
            initial,
            cfg,
            HvacCapabilities(family="fixed_speed", fingerprint="test"),
            temperature,
            np.zeros(720),
            modes,
            np.random.default_rng(17),
            cancel=cancelled,
        )
    with pytest.raises(HvacTimeout):
        simulate(
            request,
            fit,
            power,
            initial,
            cfg,
            HvacCapabilities(family="fixed_speed", fingerprint="test"),
            temperature,
            np.zeros(720),
            modes,
            np.random.default_rng(17),
            deadline=0.0,
        )


def test_probabilities_are_not_combined_from_finer_grid():
    """Disjoint path events make max/sum/independent-union reductions incorrect."""
    on = np.zeros((100, 15), bool)
    on[:50, 0] = True
    on[50:75, 5] = True
    on[:25, 10] = True
    paths = MinutePaths(ORIGIN, np.where(on, 4000.0, 100.0).astype(np.float32), on, on)
    fine = interval_stats(
        paths,
        tuple(
            GridInterval(start=ORIGIN + timedelta(minutes=i), minutes=5)
            for i in (0, 5, 10)
        ),
    )
    coarse = interval_stats(paths, (GridInterval(start=ORIGIN, minutes=15),))
    assert coarse.run_probability[0] == 0.75
    assert coarse.run_probability[0] != fine.run_probability.max()
    assert coarse.run_probability[0] != fine.run_probability.sum()
    assert coarse.run_probability[0] != pytest.approx(
        1 - np.prod(1 - fine.run_probability)
    )


def test_off_release_recovery_and_minimum_spells(setup):
    cfg, _, fit, power, request, initial = setup
    modes = np.full((cfg.n_paths, 720), "NORMAL")
    modes[:, 60:120] = "OFF"
    paths = simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        HvacCapabilities(
            family="fixed_speed",
            fingerprint="test",
            min_run_minutes=8.0,
            min_off_minutes=5.0,
        ),
        np.full(modes.shape, 7.0, np.float32),
        np.zeros(720),
        modes,
        np.random.default_rng(12),
    ).paths
    assert not paths.running[:, 60:120].any()
    assert np.all(paths.power_w[:, 60:120] == 50.0)
    assert paths.starts[:, 120:].any()
    # The first idle has the manufacturer's floor, even when the residual is zero.
    instant = InitialState(
        np.zeros(cfg.n_paths, bool),
        np.zeros(cfg.n_paths),
        np.zeros(cfg.n_paths),
        12.0,
        np.zeros(cfg.n_paths, bool),
        InputTier.T1_POWER,
        False,
    )
    locked = simulate(
        request,
        fit,
        power,
        instant,
        cfg,
        HvacCapabilities(family="fixed_speed", fingerprint="test", min_off_minutes=5.0),
        np.full(modes.shape, 7.0, np.float32),
        np.zeros(720),
        np.full(modes.shape, "NORMAL"),
        np.random.default_rng(12),
    ).paths
    assert not locked.running[:, :12].any()


def test_overload_has_no_natural_stop_and_warming_exits(setup):
    cfg, _, fit, power, request, initial = setup
    temperature = np.full((cfg.n_paths, 720), -35.0, np.float32)
    modes = np.full(temperature.shape, "NORMAL")
    continuous = simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        HvacCapabilities(family="fixed_speed", fingerprint="test"),
        temperature,
        np.zeros(720),
        modes,
        np.random.default_rng(1),
    ).paths
    assert continuous.running.all()
    temperature[:, 360:] = 7.0
    warmed = simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        HvacCapabilities(family="fixed_speed", fingerprint="test"),
        temperature,
        np.zeros(720),
        modes,
        np.random.default_rng(1),
    ).paths
    assert warmed.running[:, :360].all()
    assert not warmed.running[:, 400:].all()


def test_uncertain_off_release_is_shared_across_plan_pieces(setup):
    cfg, _, _, _, request, _ = setup
    spans = (
        ModeSpan(
            start=ORIGIN,
            end=ORIGIN + timedelta(minutes=60),
            mode="OFF",
            source="thermostat",
            end_uncertain=True,
            release_mode="NORMAL",
            elapsed_min=10.0,
        ),
        ModeSpan(
            start=ORIGIN + timedelta(minutes=60),
            end=ORIGIN + timedelta(minutes=120),
            mode="OFF",
            source="thermostat",
            end_uncertain=True,
            release_mode="ECO",
            elapsed_min=70.0,
        ),
    )
    request = HvacForecastRequest(
        ORIGIN, request.recent, spans, "OFF", pd.DataFrame(), request.grid
    )
    modes, _ = schedule_modes(
        request,
        cfg,
        pd.date_range(ORIGIN, periods=720, freq="min"),
        np.random.default_rng(1),
        comfort_off=(np.log(55.0), 0.2),
    )
    assert np.isin(modes[:, :60], ["OFF", "NORMAL"]).all()
    assert np.isin(modes[:, 60:120], ["OFF", "ECO"]).all()
    released_before = modes[:, 59] == "NORMAL"
    assert (modes[released_before, 60] == "ECO").all()


def test_defrost_power_is_measured_and_backup_is_overload_only(setup):
    """Interruptions add wall time once and retain observed compressor-on evidence."""
    from dataclasses import replace

    cfg, _, fit, power, request, initial = setup
    cfg = cfg.model_copy(update={"n_paths": 64})
    initial = replace(
        initial,
        running=initial.running[:64],
        elapsed_min=initial.elapsed_min[:64],
        remaining_min=initial.remaining_min[:64],
        water_draw=initial.water_draw[:64],
    )
    modes = np.full((64, 720), "NORMAL")
    fit = replace(fit, defrost_rate=(2.0, 2.0, 2.0), defrost_length_min=3.0)
    power = replace(power, defrost_w=700.0, defrost_running_share=1.0)
    warm = simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        HvacCapabilities(family="fixed_speed", fingerprint="test"),
        np.full(modes.shape, 7.0, np.float32),
        np.zeros(720),
        modes,
        np.random.default_rng(18),
    ).paths
    interrupt = warm.power_w == 700.0
    assert interrupt.any() and warm.running[interrupt].all()
    # No natural stop while overloaded; the backup follows overload depth.
    fit = replace(fit, defrost_rate=(0.0, 0.0, 0.0))
    cold = simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        HvacCapabilities(
            family="fixed_speed",
            fingerprint="test",
            backup_heater_kw=6.0,
            backup_heater_max_outdoor_c=-20.0,
        ),
        np.full(modes.shape, -35.0, np.float32),
        np.zeros(720),
        modes,
        np.random.default_rng(18),
    )
    assert cold.paths.power_w.min() > 4000.0
    assert "backup_heater_assumed" in cold.diagnostics


def test_continuous_overload_still_has_observed_defrost_interruptions(setup):
    from dataclasses import replace

    cfg, _, fit, power, request, initial = setup
    cfg = cfg.model_copy(update={"n_paths": 64})
    initial = replace(
        initial,
        running=initial.running[:64],
        elapsed_min=initial.elapsed_min[:64],
        remaining_min=initial.remaining_min[:64],
        water_draw=initial.water_draw[:64],
    )
    fit = replace(fit, defrost_length_min=3.0, defrost_hourly_rate=(3.0, 1.0, 0.0))
    power = replace(power, defrost_w=700.0, defrost_running_share=0.0)
    paths = simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        HvacCapabilities(family="fixed_speed", fingerprint="test"),
        np.full((64, 720), -35.0, np.float32),
        np.zeros(720),
        np.full((64, 720), "NORMAL"),
        np.random.default_rng(18),
    ).paths
    assert (paths.power_w == 700.0).any()
    assert not paths.starts.any()  # interruptions stay within the merged run
    assert not paths.running[paths.power_w == 700.0].any()
    assert paths.running[paths.power_w == 4000.0].all()
