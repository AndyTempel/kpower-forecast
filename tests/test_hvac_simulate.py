"""Simulation invariants: full ensembles, phase conditioning, commands and schedules."""

from datetime import UTC, datetime, timedelta
from threading import Event

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.aggregate import interval_stats
from kpower_forecast.hvac.cycle_model import CycleFit
from kpower_forecast.hvac.power import PowerFit
from kpower_forecast.hvac.simulate import (
    InitialState,
    ModeGrid,
    initial_state,
    schedule_modes,
    simulate,
)
from kpower_forecast.hvac.types import (
    GridInterval,
    HvacCancelled,
    HvacCapabilities,
    HvacForecastRequest,
    HvacModelConfig,
    MeteredScope,
    MinutePaths,
    ModeSpan,
)

ORIGIN = datetime(2026, 11, 9, tzinfo=UTC)
CAP = HvacCapabilities(
    family="fixed_speed", fingerprint="t", min_run_minutes=10, min_off_minutes=15
)


@pytest.fixture
def setup():
    """Constant regime: idle median 45 min, run 15 min, realistic dispersion."""
    cfg = HvacModelConfig(model_id="test", timezone="UTC", n_paths=2048)
    fit = CycleFit(
        ("ECO",),
        (np.log(45.0), 0.0, 0.0, np.log(2.0), 0.0),
        0.25,
        (np.log(15.0), 0.0, 0.0, 0.0),
        0.2,
        15.0,
        0.0,
        10.0,
        50,
        50,
    )
    power = PowerFit(
        (2000.0, *(4000.0 + 50.0 * k for k in range(1, 20))),
        0.0,
        0.0,
        100.0,
        20.0,
        MeteredScope.WHOLE_UNIT,
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
    return cfg, fit, power, request


def _run(setup, initial, modes=None, seed=3):
    cfg, fit, power, request = setup
    horizon = 720
    if modes is None:
        modes = np.full((cfg.n_paths, horizon), "NORMAL")
    return simulate(
        request,
        fit,
        power,
        initial,
        cfg,
        CAP,
        np.full((cfg.n_paths, horizon), 7.0, np.float32),
        np.zeros(horizon),
        ModeGrid.from_names(modes),
        np.random.default_rng(seed),
    )


def test_deterministic_full_ensemble_and_dephasing(setup):
    first = _run(setup, InitialState(False, 0.0, 15.0))
    again = _run(setup, InitialState(False, 0.0, 15.0))
    np.testing.assert_array_equal(first.paths.power_w, again.paths.power_w)
    running = first.paths.running
    assert running.shape == (2048, 720)
    # Just after a 15-minute run, nobody restarts before the minimum off time,
    # most have restarted near the 45-minute median, and phase is lost later.
    assert not running[:, :15].any()
    assert running[:, 25:70].any(axis=1).mean() > 0.9
    early = running[:, :120].mean(axis=0)
    late = running[:, 600:].mean(axis=0)
    assert early.std() > 3 * late.std()
    assert late.mean() == pytest.approx(0.25, abs=0.05)


def test_elapsed_idle_conditions_the_next_start(setup):
    fresh = _run(setup, InitialState(False, 5.0, 15.0)).paths.running
    old = _run(setup, InitialState(False, 40.0, 15.0)).paths.running
    first_fresh = fresh.argmax(axis=1).mean()
    first_old = old.argmax(axis=1).mean()
    assert first_old < 15 < first_fresh


def test_running_respects_minimum_run_and_ramps_power(setup):
    result = _run(setup, InitialState(True, 2.0, None))
    assert result.paths.running[:, :8].all()
    # Power keeps rising through the run until its plateau.
    p = result.paths.power_w[:, :8].mean(axis=0)
    assert np.all(np.diff(p) > 0)
    started = result.paths.starts
    assert not started[:, 0].any()


def test_off_mode_stops_runs_and_mode_change_resamples(setup):
    cfg = setup[0]
    modes = np.full((cfg.n_paths, 720), "NORMAL")
    modes[:, 30:120] = "OFF"
    result = _run(setup, InitialState(True, 2.0, None), modes=modes)
    assert not result.paths.running[:, 30:120].any()
    assert (result.paths.power_w[:, 30:120] == 20.0).all()
    # After a 90-minute off, the water is cold: almost all restart promptly.
    assert result.paths.running[:, 120:140].any(axis=1).mean() > 0.9
    eco = np.full((cfg.n_paths, 720), "ECO")
    slow = _run(setup, InitialState(False, 0.0, 15.0), modes=eco).paths.running
    normal = _run(setup, InitialState(False, 0.0, 15.0)).paths.running
    assert slow[:, :720].mean() < normal.mean()


def test_unknown_phase_is_stationary_and_cancellation_stops(setup):
    result = _run(setup, InitialState(None))
    assert result.diagnostics == ("stationary_initial_state",)
    assert result.paths.running[:, :30].mean() == pytest.approx(0.25, abs=0.06)
    cfg, fit, power, request = setup
    cancel = Event()
    cancel.set()
    with pytest.raises(HvacCancelled):
        simulate(
            request,
            fit,
            power,
            InitialState(None),
            cfg,
            CAP,
            np.full((cfg.n_paths, 720), 7.0, np.float32),
            np.zeros(720),
            ModeGrid.from_names(np.full((cfg.n_paths, 720), "NORMAL")),
            np.random.default_rng(1),
            cancel=cancel,
        )


def test_initial_state_reads_spell_age_previous_run_and_staleness():
    cfg = HvacModelConfig(model_id="t", timezone="UTC")
    index = pd.date_range(ORIGIN - timedelta(minutes=60), periods=58, freq="min")
    running = np.r_[np.zeros(20), np.ones(18), np.zeros(20)]
    recent = pd.DataFrame({"running": running}, index=index)
    state = initial_state(recent, ORIGIN, cfg)
    assert state.running is False
    assert state.elapsed_min == pytest.approx(22.0)
    assert state.prev_run_min == pytest.approx(18.0)
    stale = initial_state(recent.iloc[:-15], ORIGIN, cfg)
    assert stale.running is None


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


def test_uncertain_off_release_is_shared_across_plan_pieces(setup):
    cfg, _, _, request = setup
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
    grid, _ = schedule_modes(
        request,
        cfg,
        pd.date_range(ORIGIN, periods=720, freq="min"),
        np.random.default_rng(1),
        comfort_off=(np.log(55.0), 0.2),
    )
    modes = grid.names()
    assert np.isin(modes[:, :60], ["OFF", "NORMAL"]).all()
    assert np.isin(modes[:, 60:120], ["OFF", "ECO"]).all()
    released_before = modes[:, 59] == "NORMAL"
    assert (modes[released_before, 60] == "ECO").all()


def test_uncovered_schedule_periods_hold_the_origin_last_mode(setup):
    from dataclasses import replace

    cfg, _, _, request = setup
    span = ModeSpan(
        start=ORIGIN + timedelta(minutes=30),
        end=ORIGIN + timedelta(minutes=60),
        mode="ECO",
        source="plan",
    )
    grid, _ = schedule_modes(
        replace(request, schedule=(span,)),
        cfg,
        pd.date_range(ORIGIN, periods=120, freq="min"),
        np.random.default_rng(1),
    )
    modes = grid.names()
    assert (modes[:, :30] == "NORMAL").all()
    assert (modes[:, 30:60] == "ECO").all()
    assert (modes[:, 60:] == "NORMAL").all()


def test_automation_can_hide_a_comfort_release_without_re_latching_it(setup):
    from dataclasses import replace

    cfg, _, _, request = setup
    spans = (
        ModeSpan(
            start=ORIGIN,
            end=ORIGIN + timedelta(minutes=17),
            mode="BOOST",
            source="automation",
        ),
        ModeSpan(
            start=ORIGIN + timedelta(minutes=17),
            end=ORIGIN + timedelta(minutes=60),
            mode="OFF",
            source="thermostat",
            end_uncertain=True,
            elapsed_min=47.0,
            release_mode="NORMAL",
        ),
    )
    grid, _ = schedule_modes(
        replace(request, schedule=spans),
        cfg,
        pd.date_range(ORIGIN, periods=120, freq="min"),
        np.random.default_rng(1),
        comfort_off=(np.log(35.0), 0.01),
    )
    modes = grid.names()
    assert (modes[:, :17] == "BOOST").all()
    assert (
        modes[:, 17:60] == "NORMAL"
    ).all()  # release sampled at origin, age30, not age47
