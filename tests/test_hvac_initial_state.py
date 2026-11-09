"""Heat-pump runs forecast from the water temperatures (R1b)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from math import pi, sin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.initial_state import (
    MINUTE,
    OUTDOOR_COLUMN,
    OUTDOOR_INVALID_COLUMN,
    POWER_COLUMN,
    RETURN_COLUMN,
    SUPPLY_COLUMN,
    fit_water_cycles,
    project_water_runs,
)

STEP = timedelta(minutes=5)
START = datetime(2026, 10, 1, tzinfo=UTC)
RUN_W, IDLE_W, THRESHOLD_W = 4700.0, 95.0, 2400.0
TRIGGERS = {"NORMAL": 26.0, "ECO": 24.0}
COOLING_TO_C, COOLING = 21.0, 0.023
RUN_MINUTES = 20


def _outdoor(minute: int, swing_c: float) -> float:
    return 8.0 + swing_c * sin(2 * pi * minute / (3 * 1440))


def _history(
    minutes: int,
    *,
    ending_idle_c: float | None = None,
    outdoor_slope: float = 0.0,
    outdoor_swing_c: float = 6.0,
    defrost_every: int = 0,
) -> tuple[pd.DataFrame, list[tuple[datetime, str]]]:
    """A heat pump cycling on its return water: starts at the trigger, runs 20 min.

    The site goes to ECO overnight, so each mode's trigger is seen. With
    ``ending_idle_c`` the history stops idle, at that return water. With
    ``outdoor_slope`` the trigger follows the outdoor temperature. With
    ``defrost_every`` every n-th run stops and restarts a minute later, warm.
    """
    commands: list[tuple[datetime, str]] = [(START - MINUTE, "NORMAL")]
    power: list[float] = []
    returns: list[float] = []
    supplies: list[float] = []
    temperature, run_left, current = 30.0, 0, "NORMAL"
    runs, restart_in, defrosted = 0, 0, 0
    for minute in range(minutes):
        at = START + MINUTE * minute
        wanted = "ECO" if at.hour < 6 else "NORMAL"
        if wanted != current and run_left == 0:
            current = wanted
            commands.append((at - timedelta(minutes=10), current))
        outdoor = _outdoor(minute, outdoor_swing_c)
        trigger = TRIGGERS[current] + outdoor_slope * (outdoor - 8.0)
        if restart_in > 0:
            restart_in -= 1
            if restart_in == 0:
                run_left = RUN_MINUTES
        elif run_left == 0 and temperature <= trigger:
            run_left = RUN_MINUTES
            runs += 1
        if run_left > 0:
            run_left -= 1
            if (
                defrost_every
                and runs % defrost_every == 0
                and run_left == 8
                and defrosted != runs
            ):
                run_left, restart_in, defrosted = 0, 2, runs
            temperature += 0.6
            power.append(RUN_W)
            supplies.append(temperature + 5.0)
        else:
            temperature -= COOLING * (temperature - COOLING_TO_C)
            power.append(IDLE_W)
            supplies.append(temperature + 0.3)
        returns.append(temperature)
    index = pd.date_range(START, periods=minutes, freq="1min")
    frame = pd.DataFrame(
        {POWER_COLUMN: power, RETURN_COLUMN: returns, SUPPLY_COLUMN: supplies},
        index=index,
    )
    if outdoor_slope:
        frame[OUTDOOR_COLUMN] = [
            _outdoor(minute, outdoor_swing_c) for minute in range(minutes)
        ]
    if ending_idle_c is not None:
        idle = frame[frame[POWER_COLUMN] < THRESHOLD_W]
        last = idle[(idle[RETURN_COLUMN] - ending_idle_c).abs() < 0.1].index[-1]
        frame = frame[frame.index <= last]
    commands = [item for item in commands if item[0] <= frame.index[-1]]
    return frame, commands


def _project(
    frame: pd.DataFrame,
    commands: list[tuple[datetime, str]],
    planned: list[str | None],
    *,
    mode_now: str = "NORMAL",
) -> list[float]:
    now = frame.index[-1].to_pydatetime() + MINUTE
    cycles, reason = fit_water_cycles(frame, commands, threshold_w=THRESHOLD_W, now=now)
    assert reason is None and cycles is not None
    base = now.replace(second=0, minute=now.minute // 5 * 5)
    probabilities, why = project_water_runs(
        cycles,
        latest=frame,
        mode_now=mode_now,
        step_starts=[base + STEP * index for index in range(len(planned))],
        planned_modes=planned,
        stop_share={},
        threshold_w=THRESHOLD_W,
        idle_w=IDLE_W,
        run_w=RUN_W,
        now=now,
    )
    assert why is None and probabilities is not None
    return probabilities


def test_cooling_and_the_trigger_per_mode_are_learned() -> None:
    frame, commands = _history(10 * 1440)
    cycles, reason = fit_water_cycles(
        frame, commands, threshold_w=THRESHOLD_W, now=frame.index[-1].to_pydatetime()
    )
    assert reason is None and cycles is not None
    assert cycles.cooling_per_min == pytest.approx(COOLING, rel=0.05)
    assert cycles.cooling_to_c == pytest.approx(COOLING_TO_C, abs=0.5)
    for mode, trigger in TRIGGERS.items():
        assert cycles.trigger_c(mode) == pytest.approx(trigger, abs=0.3)


def _fitted(frame: pd.DataFrame, commands: list):  # type: ignore[no-untyped-def]
    cycles, reason = fit_water_cycles(
        frame,
        commands,
        threshold_w=THRESHOLD_W,
        now=frame.index[-1].to_pydatetime() + MINUTE,
    )
    assert reason is None and cycles is not None
    return cycles


def test_the_trigger_follows_the_outdoor_temperature() -> None:
    frame, commands = _history(14 * 1440, outdoor_slope=-0.3)
    cycles = _fitted(frame, commands)
    line = cycles.trigger_line
    assert line is not None
    assert line.slope == pytest.approx(-0.3, abs=0.05)
    assert line.levels_c["NORMAL"] - line.levels_c["ECO"] == pytest.approx(2.0, abs=0.3)
    expected = TRIGGERS["NORMAL"] - 0.3 * (cycles.outdoor_c - 8.0)
    assert cycles.trigger_c("NORMAL") == pytest.approx(expected, abs=0.3)
    # It does not extrapolate beyond the outdoor range it has seen.
    assert line.trigger_c("NORMAL", -40.0) == line.trigger_c(
        "NORMAL", line.outdoor_low_c
    )


def test_without_usable_outdoor_the_latest_starts_set_the_trigger() -> None:
    frame, commands = _history(14 * 1440, outdoor_slope=-0.3)
    stale = frame.copy()
    stale.loc[stale.index[-120:], OUTDOOR_COLUMN] = float("nan")
    assert _fitted(stale, commands).trigger_line is None

    # Too little outdoor range for a slope to mean anything.
    flat, commands = _history(14 * 1440, outdoor_slope=-0.3, outdoor_swing_c=1.0)
    assert _fitted(flat, commands).trigger_line is None

    # A trigger that rises with the outdoor temperature is not believed.
    rising, commands = _history(14 * 1440, outdoor_slope=0.3)
    assert _fitted(rising, commands).trigger_line is None


def test_a_brief_outdoor_outage_is_tolerated_and_a_long_one_is_not() -> None:
    frame, commands = _history(14 * 1440, outdoor_slope=-0.3)
    frame = frame.copy()
    # Silent, as in an outage: the last reading was 3 minutes before the end.
    frame.loc[frame.index[-3:], OUTDOOR_COLUMN] = float("nan")
    frame[OUTDOOR_INVALID_COLUMN] = float("nan")
    frame.loc[frame.index[-2], OUTDOOR_INVALID_COLUMN] = 1.0
    assert _fitted(frame, commands).trigger_line is not None

    frame[OUTDOOR_INVALID_COLUMN] = float("nan")
    frame.loc[frame.index[-40:], OUTDOOR_COLUMN] = float("nan")
    frame.loc[frame.index[-20], OUTDOOR_INVALID_COLUMN] = 1.0
    assert _fitted(frame, commands).trigger_line is None

    # A valid reading after the outage counts again.
    frame.loc[frame.index[-1], OUTDOOR_COLUMN] = 7.0
    cycles = _fitted(frame, commands)
    assert cycles.trigger_line is not None and cycles.outdoor_c == 7.0


def test_a_telemetry_blip_does_not_lose_a_start_or_a_runs_age() -> None:
    frame, commands = _while_running(minutes_in=10)
    on = frame[POWER_COLUMN] >= THRESHOLD_W
    begun = frame.index[on & ~on.shift(1, fill_value=False)]
    whole = _fitted(frame, commands)
    # Minutes lost around a start and inside the current run.
    holes = [begun[-5] - MINUTE * 2, begun[-5] - MINUTE, begun[-1] + MINUTE * 4]
    blipped = _fitted(frame.drop(holes), commands)
    assert len(blipped.starts) == len(whole.starts)
    assert blipped.running_since == whole.running_since


def test_each_minute_counts_in_the_step_that_holds_it() -> None:
    frame, commands = _while_running(minutes_in=RUN_MINUTES - 4)
    # 3 running minutes remain after the latest measured one.
    now = frame.index[-1].to_pydatetime() + MINUTE
    probabilities, _ = project_water_runs(
        _fitted(frame, commands),
        latest=frame,
        mode_now="NORMAL",
        step_starts=[now, now + STEP],
        planned_modes=["NORMAL", "NORMAL"],
        stop_share={},
        threshold_w=IDLE_W + (RUN_W - IDLE_W) * 0.5,
        idle_w=IDLE_W,
        run_w=RUN_W,
        now=now,
    )
    assert probabilities is not None
    # 3 of the first step's 5 minutes run (0.6 ≥ 0.5), none of the second's.
    assert probabilities == [1.0, 0.0]


def test_a_restart_after_a_stop_says_nothing_about_the_trigger() -> None:
    frame, commands = _history(14 * 1440, defrost_every=3)
    cycles = _fitted(frame, commands)
    assert max(value for _, value in cycles.starts) < 27.0
    assert cycles.trigger_c("NORMAL") == pytest.approx(TRIGGERS["NORMAL"], abs=0.3)


def test_a_start_is_forecast_when_the_water_will_reach_the_trigger() -> None:
    frame, commands = _history(10 * 1440, ending_idle_c=29.0)
    # From 29 °C the return water reaches 26 °C in about 20 minutes.
    probabilities = _project(frame, commands, ["NORMAL"] * 12)
    assert max(probabilities[:2]) < 0.1
    assert max(probabilities[5:8]) > 0.9


def test_off_holds_a_start_back_and_its_release_starts_a_run() -> None:
    frame, commands = _history(10 * 1440, ending_idle_c=27.0)
    planned: list[str | None] = ["OFF"] * 6 + ["NORMAL"] * 6
    commands = [*commands, (frame.index[-1].to_pydatetime(), "OFF")]
    probabilities = _project(frame, commands, planned, mode_now="OFF")
    assert probabilities[:6] == [0.0] * 6
    # The water passed the trigger under OFF, so the release runs at once.
    assert probabilities[6] > 0.9


def test_a_mode_with_a_higher_trigger_starts_sooner() -> None:
    frame, commands = _history(10 * 1440, ending_idle_c=25.5)
    eco = _project(frame, commands, ["ECO"] * 3, mode_now="ECO")
    normal = _project(frame, commands, ["NORMAL"] * 3, mode_now="ECO")
    assert eco[0] < 0.1
    assert normal[1] > 0.9


def _while_running(minutes_in: int = 5) -> tuple[pd.DataFrame, list]:
    frame, commands = _history(10 * 1440)
    on = frame[POWER_COLUMN] >= THRESHOLD_W
    begun = frame.index[on & ~on.shift(1, fill_value=False)]
    frame = frame[frame.index <= begun[-1] + MINUTE * minutes_in]
    return frame, [item for item in commands if item[0] <= frame.index[-1]]


def test_a_long_off_counts_toward_the_idle_gap_and_its_release_starts_at_once() -> None:
    frame, commands = _while_running()
    planned: list[str | None] = ["OFF"] * 12 + ["NORMAL"] * 2
    # The OFF is the plan's, and it cuts the run: nothing runs under it.
    probabilities = _project(frame, commands, planned)
    assert max(probabilities[2:12]) == 0.0
    assert probabilities[12] > 0.9


def test_a_mode_change_in_the_first_minute_still_stops_a_run() -> None:
    frame, commands = _while_running()
    now = frame.index[-1].to_pydatetime() + MINUTE
    cycles, _ = fit_water_cycles(frame, commands, threshold_w=THRESHOLD_W, now=now)
    assert cycles is not None
    base = now.replace(second=0, minute=now.minute // 5 * 5)
    probabilities, _ = project_water_runs(
        cycles,
        latest=frame,
        mode_now="NORMAL",
        step_starts=[base + STEP * index for index in range(4)],
        planned_modes=["ECO"] * 4,
        stop_share={("NORMAL", "ECO"): 1.0},
        threshold_w=THRESHOLD_W,
        idle_w=IDLE_W,
        run_w=RUN_W,
        now=now,
    )
    assert probabilities is not None
    assert probabilities[2] == 0.0


def test_an_off_the_plan_does_not_explain_is_left_to_the_hazard() -> None:
    frame, commands = _history(10 * 1440)
    now = frame.index[-1].to_pydatetime() + MINUTE
    cycles, _ = fit_water_cycles(frame, commands, threshold_w=THRESHOLD_W, now=now)
    assert cycles is not None
    probabilities, reason = project_water_runs(
        cycles,
        latest=frame,
        mode_now="OFF",
        step_starts=[now],
        planned_modes=["NORMAL"],
        stop_share={},
        threshold_w=THRESHOLD_W,
        idle_w=IDLE_W,
        run_w=RUN_W,
        now=now,
    )
    assert probabilities is None and reason == "unexplained_off"


def test_no_water_temperatures_give_no_cycles() -> None:
    frame, commands = _history(3 * 1440)
    frame[RETURN_COLUMN] = float("nan")
    cycles, reason = fit_water_cycles(
        frame, commands, threshold_w=THRESHOLD_W, now=frame.index[-1].to_pydatetime()
    )
    assert cycles is None and reason == "no_water_history"


@pytest.mark.parametrize("case_number", [0, 1, 2])
def test_unchanged_ems_numerical_golden_parity_to_one_nanounit(case_number):
    from kpower_forecast.hvac.initial_state import TRIGGER_QUANTILES, _Member, _play

    golden = json.loads(
        (Path(__file__).parent / "fixtures/hvac_water_parity.json").read_text()
    )
    case = golden["cases"][case_number]
    frame, commands = _history(14 * 1440, **case["kwargs"])
    cycles = _fitted(frame, commands)
    assert cycles.cooling_per_min == pytest.approx(case["cooling_per_min"], abs=1e-9)
    assert cycles.cooling_to_c == pytest.approx(case["cooling_to_c"], abs=1e-9)
    assert cycles.trigger_c("NORMAL") == pytest.approx(
        case["trigger_normal_c"], abs=1e-9
    )
    assert cycles.trigger_c("ECO") == pytest.approx(case["trigger_eco_c"], abs=1e-9)
    np.testing.assert_allclose(
        _project(frame, commands, case["planned"]),
        case["probabilities"],
        atol=1e-9,
        rtol=0,
    )
    times = []
    for error in np.quantile(cycles.spread_c, TRIGGER_QUANTILES):
        played = _play(
            cycles,
            _Member(float(error), 20.0, 45.0, 20.0, 0.0),
            return_c=float(frame[RETURN_COLUMN].iloc[-1]),
            running=False,
            mode_now="NORMAL",
            modes=["NORMAL"] * 240,
            triggers={"NORMAL": cycles.trigger_c("NORMAL")},
            stop_share={},
        )
        where = np.flatnonzero(played)
        times.append(int(where[0]) if len(where) else None)
    assert times == case["first_start_minutes"]
