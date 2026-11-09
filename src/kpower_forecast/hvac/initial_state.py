"""Verbatim R1c numerical water-trigger port from EMS.

The numerical functions and constants retain their source behavior. Transport
column labels and two pure timestamp/command helpers are local so this module
imports without EMS or any forecasting backend. Mixture wiring comes separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._stats import conditional_lognormal
from .cycle_model import CycleParameters
from .types import HvacModelConfig, InputTier

# Transport-neutral legacy column labels retained for numerical parity.
WATER_CYCLE_POWER_KEY = "power_w"
WATER_CYCLE_RETURN_KEY = "return_c"
WATER_CYCLE_SUPPLY_KEY = "supply_c"
WATER_CYCLE_OUTDOOR_KEY = "outdoor_c"
WATER_CYCLE_OUTDOOR_INVALID_KEY = "outdoor_invalid_at"


def _mode_changes(
    commands: Sequence[tuple[datetime, str]],
) -> list[tuple[datetime, str]]:
    """Commands that changed the mode; a re-issued mode is not a change."""
    changes: list[tuple[datetime, str]] = []
    for at, mode in sorted(commands, key=lambda item: item[0]):
        if not changes or changes[-1][1] != mode:
            changes.append((at, mode))
    return changes


def _parse_utc(value: object) -> datetime | None:
    """A UTC timestamp, or ``None`` when ``value`` is not one."""
    if isinstance(value, datetime):
        stamp = pd.Timestamp(value)
    elif isinstance(value, str) and value:
        try:
            stamp = pd.Timestamp(value)
        except ValueError:
            return None
    else:
        return None
    if pd.isna(stamp):
        return None
    stamp = (
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    )
    return stamp.to_pydatetime()


MINUTE = timedelta(minutes=1)
_STEP_MINUTES = 5
MAX_MINUTE_AGE = timedelta(minutes=10)
# A command this soon before a start may have caused it, so it says nothing
# about the trigger.
START_COMMAND_WINDOW = timedelta(minutes=5)
# A start this soon after a stop restarts an interrupted run, with warm water.
MIN_RESTART_IDLE = timedelta(minutes=15)
# Natural starts behind a trigger. Of 2, 3 and 5, 3 scored best on the tuning
# span of the replay (2026-10-21 to 11-01); the later days were kept out.
TRIGGER_STARTS = 3
MIN_TRIGGER_STARTS = 2
MIN_WATER_STARTS = 20
# An outdoor reading older than this says nothing about now.
OUTDOOR_HOLD = timedelta(hours=1)
# An explicit invalidation ends a held reading after this, not at once. A Home
# Assistant update, a restart or a network blip must not throw the trigger back
# to the last starts, and the outdoor temperature barely moves meanwhile. The
# live sensor's longest of 6 outages in a week lasted 5.3 minutes (2026-11-06).
OUTDOOR_INVALID_GRACE = timedelta(minutes=15)
# Telemetry missing for this long still counts as one spell, so a restart or
# network blip does not lose a start. The live site's longest gap in 21 days
# was 4 minutes (2026-11-06).
MINUTE_GAP_TOLERANCE = timedelta(minutes=5)
MIN_MODE_STARTS = 5
# Below this outdoor range (10th to 90th percentile) a slope is mostly noise.
MIN_OUTDOOR_RANGE_C = 3.0
OUTLIER_SIGMAS = 2.0
# Just after a run the water mixes with the circuit and falls faster.
COOLING_BELOW_C = 36.0
MIN_COOLING_PER_MIN = 0.005
MAX_COOLING_PER_MIN = 0.1
MAX_SPREAD_STARTS = 120
MIN_RUN_MINUTES = 4.0
RECENT_RUNS = 20
MIN_RECENT_RUNS = 5
RECENT_GAPS = 8
MIN_RECENT_GAPS = 3
MAX_GAP_MINUTES = 120.0
STOP_TEMPERATURES = 6
MIN_STOP_TEMPERATURES = 3
# Weight of this projection against the hazard model's, which carries the
# average effect of unplanned commands. Alone it is overconfident: its 0.97 ran
# 0.79 of the time. Of 1, 0.9, 0.8, 0.7 and 0.6, 0.7 and 0.8 tied at the shorter
# leads on the tuning span, and 0.7 kept better calibration at the longest.
WATER_WEIGHT = 0.7
# Minutes of a run before the supply water's rise says when it will end.
RISE_AFTER_MINUTES = 6.0
RISE_WINDOW = timedelta(minutes=5)
MIN_RISE_PER_MIN = 0.1
TRIGGER_QUANTILES = tuple(float(q) for q in np.linspace(0.05, 0.95, 13))
RUN_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
GAP_QUANTILES = (0.15, 0.5, 0.85)
_OFF = "OFF"

POWER_COLUMN = WATER_CYCLE_POWER_KEY
RETURN_COLUMN = WATER_CYCLE_RETURN_KEY
SUPPLY_COLUMN = WATER_CYCLE_SUPPLY_KEY
OUTDOOR_COLUMN = WATER_CYCLE_OUTDOOR_KEY
# 1.0 at each moment the outdoor sensor stopped being valid.
OUTDOOR_INVALID_COLUMN = WATER_CYCLE_OUTDOOR_INVALID_KEY

# Spreads the members' ranks evenly, whatever order they are built in.
_RANK_STEP = (5.0**0.5 - 1.0) / 2.0

Command = tuple[datetime, str]


@dataclass(frozen=True, slots=True)
class TriggerLine:
    """The return water at which runs start, against the outdoor temperature."""

    levels_c: Mapping[str, float]
    slope: float
    # The outdoor range it was fitted on; it is not extrapolated beyond that.
    outdoor_low_c: float
    outdoor_high_c: float

    def trigger_c(self, mode: str, outdoor_c: float) -> float | None:
        level = self.levels_c.get(mode)
        if level is None:
            return None
        return level + self.slope * min(
            max(outdoor_c, self.outdoor_low_c), self.outdoor_high_c
        )


@dataclass(frozen=True, slots=True)
class WaterCycles:
    """What the site's water temperatures say about its cycle."""

    cooling_per_min: float
    cooling_to_c: float
    # ``(mode, return water)`` of the natural starts, oldest first.
    starts: tuple[tuple[str, float], ...]
    # Return water at a natural start, minus the trigger known at the time.
    spread_c: tuple[float, ...]
    # ``(mode at the end, minutes, supply water)`` of runs no command touched.
    runs: tuple[tuple[str, float, float], ...]
    gaps_min: tuple[float, ...]
    # When the running compressor started, if it is running.
    running_since: datetime | None
    trigger_line: TriggerLine | None = None
    outdoor_c: float | None = None

    def trigger_c(self, mode: str) -> float | None:
        """The return water at which a run starts in this mode."""
        if self.trigger_line is not None and self.outdoor_c is not None:
            trigger = self.trigger_line.trigger_c(mode, self.outdoor_c)
            if trigger is not None:
                return trigger
        same = [value for at_mode, value in self.starts if at_mode == mode]
        if len(same) < MIN_TRIGGER_STARTS:
            same = [value for _, value in self.starts]
        recent = same[-TRIGGER_STARTS:]
        return float(np.median(recent)) if len(recent) >= MIN_TRIGGER_STARTS else None


def minute_frame(series: Mapping[str, Any]) -> pd.DataFrame:
    """One DataFrame from the data-storage water-cycle payload (UTC, by minute)."""
    columns: dict[str, pd.Series] = {}
    for column in (POWER_COLUMN, RETURN_COLUMN, SUPPLY_COLUMN, OUTDOOR_COLUMN):
        rows = series.get(column)
        values: dict[pd.Timestamp, float] = {}
        for row in rows if isinstance(rows, list) else []:
            if not (isinstance(row, list) and len(row) == 2):
                continue
            stamp, value = _parse_utc(row[0]), row[1]
            if stamp is not None and isinstance(value, int | float):
                values[pd.Timestamp(stamp)] = float(value)
        columns[column] = pd.Series(
            list(values.values()),
            index=pd.to_datetime(list(values), utc=True),
            dtype=float,
        )
    invalid_rows = series.get(OUTDOOR_INVALID_COLUMN)
    invalid = [
        pd.Timestamp(at)
        for at in map(
            _parse_utc, invalid_rows if isinstance(invalid_rows, list) else []
        )
        if at is not None
    ]
    columns[OUTDOOR_INVALID_COLUMN] = pd.Series(
        1.0, index=pd.to_datetime(invalid, utc=True), dtype=float
    )
    return pd.DataFrame(columns).sort_index()


def _mode_at(changes: Sequence[Command], at: datetime) -> str | None:
    mode: str | None = None
    for changed, value in changes:
        if changed > at:
            break
        mode = value
    return mode


def fit_water_cycles(
    frame: pd.DataFrame,
    commands: Sequence[Command],
    *,
    threshold_w: float,
    now: datetime,
) -> tuple[WaterCycles | None, str | None]:
    """Learn the cooling, the triggers and the lengths; causal in ``now``.

    Args:
        frame: One-minute power and water temperatures (see :func:`minute_frame`).
        commands: ``(time, mode)`` of the recorded mode commands.
        threshold_w: Power at or above which the compressor is running.
        now: Fit time; later rows are ignored.

    Returns:
        The cycles, or ``None`` and the reason there are none.
    """
    needed = {POWER_COLUMN, RETURN_COLUMN, SUPPLY_COLUMN}
    if frame.empty or not needed <= set(frame.columns):
        return None, "no_water_history"
    frame = frame[frame.index <= pd.Timestamp(now)]
    outdoor = (
        frame[OUTDOOR_COLUMN].dropna()
        if OUTDOOR_COLUMN in frame.columns
        else pd.Series(dtype=float)
    )
    invalid = cast(
        pd.DatetimeIndex,
        (
            frame[OUTDOOR_INVALID_COLUMN].dropna().index
            if OUTDOOR_INVALID_COLUMN in frame.columns
            else pd.DatetimeIndex([], tz="UTC")
        ),
    )
    frame = frame.dropna(subset=[POWER_COLUMN, RETURN_COLUMN])
    if frame.empty:
        return None, "no_water_history"
    on = (frame[POWER_COLUMN] >= threshold_w).to_numpy()
    returns = frame[RETURN_COLUMN].to_numpy()
    supplies = frame[SUPPLY_COLUMN].to_numpy()
    stamps = cast(pd.DatetimeIndex, frame.index)
    steps = stamps[1:] - stamps[:-1]
    adjacent = np.asarray(steps == pd.Timedelta(MINUTE))
    contiguous = np.asarray(steps <= pd.Timedelta(MINUTE_GAP_TOLERANCE))

    cooling_mask = (~on[:-1]) & (~on[1:]) & adjacent & (returns[:-1] < COOLING_BELOW_C)
    if int(cooling_mask.sum()) < 2:
        return None, "no_water_cooling"
    slope, intercept = np.polyfit(
        returns[:-1][cooling_mask],
        (returns[1:] - returns[:-1])[cooling_mask],
        1,
    )
    cooling = -float(slope)
    if not MIN_COOLING_PER_MIN <= cooling <= MAX_COOLING_PER_MIN:
        return None, "water_cooling_unstable"
    cooling_to = float(intercept) / cooling

    changes = _mode_changes(commands)
    change_times = [at for at, _ in changes]
    starts_at = [
        index
        for index in range(1, len(on))
        if on[index] and not on[index - 1] and contiguous[index - 1]
    ]
    natural: list[tuple[str, float]] = []
    against_outdoor: list[tuple[str, float, float]] = []
    spread: list[float] = []
    runs: list[tuple[str, float, float]] = []
    gaps: list[float] = []
    previous_end: pd.Timestamp | None = None
    for start in starts_at:
        began = stamps[start].to_pydatetime()
        before = stamps[start - 1]
        end = start
        while end < len(on) and on[end]:
            end += 1
        restart = False
        if previous_end is not None:
            gap = (stamps[start] - previous_end) / pd.Timedelta(MINUTE)
            restart = gap < MIN_RESTART_IDLE / MINUTE
            if not restart and gap <= MAX_GAP_MINUTES:
                gaps.append(float(gap))
        if end >= len(on):
            previous_end = None
            continue
        ended = stamps[end].to_pydatetime()
        previous_end = stamps[end]
        minutes = (stamps[end] - stamps[start]) / pd.Timedelta(MINUTE)
        if minutes < MIN_RUN_MINUTES:
            continue
        mode = _mode_at(changes, began)
        caused = any(began - START_COMMAND_WINDOW <= at <= began for at in change_times)
        if mode is not None and mode != _OFF and not caused and not restart:
            value = float(frame[RETURN_COLUMN].get(before, np.nan))
            if np.isfinite(value):
                known = WaterCycles(
                    cooling, cooling_to, tuple(natural), (), (), (), None
                ).trigger_c(mode)
                if known is not None:
                    spread.append(value - known)
                natural.append((mode, value))
                outdoor_then = _outdoor_at(outdoor, invalid, began)
                if outdoor_then is not None:
                    against_outdoor.append((mode, outdoor_then, value))
        cut = any(began < at <= ended for at in change_times)
        end_mode = _mode_at(changes, ended)
        supply = float(supplies[end - 1])
        if not cut and end_mode is not None and np.isfinite(supply):
            runs.append((end_mode, float(minutes), supply))

    if len(natural) < MIN_WATER_STARTS:
        return None, "too_few_water_starts"
    cycles = WaterCycles(
        cooling_per_min=cooling,
        cooling_to_c=cooling_to,
        starts=tuple(natural),
        spread_c=tuple(spread[-MAX_SPREAD_STARTS:]),
        runs=tuple(runs),
        gaps_min=tuple(gaps),
        running_since=(
            _running_since(on, stamps, contiguous) if bool(on[-1]) else None
        ),
    )
    outdoor_now = _outdoor_at(outdoor, invalid, now)
    fitted = _fit_trigger_line(against_outdoor) if outdoor_now is not None else None
    if fitted is None:
        return cycles, None
    line, residuals = fitted
    return (
        replace(
            cycles,
            trigger_line=line,
            outdoor_c=outdoor_now,
            spread_c=residuals[-MAX_SPREAD_STARTS:],
        ),
        None,
    )


def _outdoor_at(
    outdoor: pd.Series, invalid: pd.DatetimeIndex, at: datetime
) -> float | None:
    """The latest outdoor reading at ``at``, if it can still be believed.

    It is too old past ``OUTDOOR_HOLD``, and it ends ``OUTDOOR_INVALID_GRACE``
    after the sensor was declared invalid.
    """
    stamp = pd.Timestamp(at)
    position = int(outdoor.index.searchsorted(stamp, side="right")) - 1
    if position < 0:
        return None
    read_at = outdoor.index[position]
    if stamp - read_at > OUTDOOR_HOLD:
        return None
    first_invalid = int(invalid.searchsorted(read_at, side="right"))
    if (
        first_invalid < len(invalid)
        and invalid[first_invalid] <= stamp
        and stamp - invalid[first_invalid] > OUTDOOR_INVALID_GRACE
    ):
        return None
    return float(outdoor.iloc[position])


def _fit_trigger_line(
    starts: Sequence[tuple[str, float, float]],
) -> tuple[TriggerLine, tuple[float, ...]] | None:
    """Fit ``(mode, outdoor, return water)``; ``None`` when no line can be trusted.

    Returns the line and the residuals of its starts, oldest first. A start
    far off the line is left out of the fit (a hot-water run, a stale reading)
    but stays in the residuals, because the forecast meets those too.
    """
    counts: dict[str, int] = {}
    for mode, _, _ in starts:
        counts[mode] = counts.get(mode, 0) + 1
    modes = [mode for mode, count in counts.items() if count >= MIN_MODE_STARTS]
    used = [start for start in starts if start[0] in modes]
    if len(used) < MIN_WATER_STARTS:
        return None
    outdoors = np.array([start[1] for start in used])
    low, high = np.quantile(outdoors, (0.1, 0.9))
    if high - low < MIN_OUTDOOR_RANGE_C:
        return None
    values = np.array([start[2] for start in used])
    design = np.column_stack(
        [
            *[[1.0 if start[0] == mode else 0.0 for start in used] for mode in modes],
            outdoors,
        ]
    )
    weights = np.linalg.lstsq(design, values, rcond=None)[0]
    errors = values - design @ weights
    keep = np.abs(errors) <= OUTLIER_SIGMAS * errors.std()
    weights = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
    slope = float(weights[-1])
    if slope > 0.0:
        # The return target does not rise with the outdoor temperature: the
        # readings do not describe this heat pump's control.
        return None
    return (
        TriggerLine(
            levels_c={mode: float(weights[i]) for i, mode in enumerate(modes)},
            slope=slope,
            outdoor_low_c=float(outdoors.min()),
            outdoor_high_c=float(outdoors.max()),
        ),
        tuple(float(value) for value in values - design @ weights),
    )


def _running_since(
    on: np.ndarray, stamps: pd.DatetimeIndex, contiguous: np.ndarray
) -> datetime | None:
    index = len(on) - 1
    while index > 0 and on[index - 1] and contiguous[index - 1]:
        index -= 1
    return stamps[index].to_pydatetime() if index > 0 else None


@dataclass(frozen=True, slots=True)
class _Member:
    """One way the next hour can go."""

    trigger_error_c: float
    # Minutes the first run lasts from the projection's start.
    run_minutes: float
    gap_minutes: float
    next_run_minutes: float
    # Where this member falls among the others, for a command that stops a share.
    rank: float


def _play(
    cycles: WaterCycles,
    member: _Member,
    *,
    return_c: float,
    running: bool,
    mode_now: str,
    modes: Sequence[str],
    triggers: Mapping[str, float | None],
    stop_share: Mapping[tuple[str, str], float],
) -> np.ndarray:
    """One member, minute by minute: whether the compressor runs."""
    out = np.zeros(len(modes), dtype=bool)
    run_left = member.run_minutes if running else 0.0
    idle_left: float | None = None
    temperature = return_c
    for minute, mode in enumerate(modes):
        before = modes[minute - 1] if minute else mode_now
        if run_left > 0.0:
            cut = mode != before and member.rank < stop_share.get((before, mode), 0.0)
            if mode == _OFF or cut:
                run_left = 0.0
                idle_left = member.gap_minutes
            else:
                out[minute] = True
                run_left -= 1.0
                if run_left <= 0.0:
                    idle_left = member.gap_minutes
                continue
        if idle_left is not None:
            # A later cycle has no temperature to go by yet: its idle spell is
            # as long as the recent ones. OFF counts toward it and holds the
            # start, so a release after a long OFF starts at once.
            idle_left -= 1.0
            if idle_left <= 0.0 and mode != _OFF:
                run_left = member.next_run_minutes
                idle_left = None
            continue
        if mode == _OFF:
            temperature = _cooled(cycles, temperature)
            continue
        trigger = triggers[mode]
        if trigger is not None and temperature <= trigger + member.trigger_error_c:
            run_left = member.run_minutes
            out[minute] = True
            run_left -= 1.0
            continue
        temperature = _cooled(cycles, temperature)
    return out


def _cooled(cycles: WaterCycles, temperature: float) -> float:
    return temperature - cycles.cooling_per_min * (temperature - cycles.cooling_to_c)


def project_water_runs(
    cycles: WaterCycles,
    *,
    latest: pd.DataFrame,
    mode_now: str | None,
    step_starts: Sequence[datetime],
    planned_modes: Sequence[str | None],
    stop_share: Mapping[tuple[str, str], float],
    threshold_w: float,
    idle_w: float,
    run_w: float,
    now: datetime,
) -> tuple[list[float] | None, str | None]:
    """The chance of a running compressor per 5-minute step.

    Args:
        cycles: The learned cycles.
        latest: Recent one-minute rows, at least the last few minutes.
        mode_now: The mode in force.
        step_starts: The near-term steps (5 minutes apart).
        planned_modes: The planned mode per step; ``None`` keeps the mode.
        stop_share: Share of runs each mode change ends, per ``(from, to)``.
        threshold_w: The 5-minute mean at or above which a step counts as running.
        idle_w: Idle power, for that threshold in on-minutes.
        run_w: Run power, for that threshold in on-minutes.
        now: Projection time.

    Returns:
        One probability per step, or ``None`` and why the hazard must answer.
    """
    rows = latest.dropna(subset=[POWER_COLUMN, RETURN_COLUMN])
    rows = rows[rows.index <= pd.Timestamp(now)]
    if rows.empty or now - rows.index[-1].to_pydatetime() > MAX_MINUTE_AGE:
        return None, "stale_water"
    if mode_now is None:
        return None, "no_mode"
    plan_now = planned_modes[0] if planned_modes else None
    if mode_now == _OFF and plan_now != _OFF:
        # An OFF the plan does not explain has an unknown release: the hazard
        # model draws it from the learned OFF lengths.
        return None, "unexplained_off"
    last = rows.iloc[-1]
    stamp = rows.index[-1].to_pydatetime()
    running = cycles.running_since is not None

    lengths = [run[1] for run in cycles.runs if run[0] == mode_now][-RECENT_RUNS:]
    if len(lengths) < MIN_RECENT_RUNS:
        lengths = [run[1] for run in cycles.runs][-RECENT_RUNS:]
    gaps = list(cycles.gaps_min[-RECENT_GAPS:])
    if (
        len(lengths) < MIN_RECENT_RUNS
        or len(gaps) < MIN_RECENT_GAPS
        or len(cycles.spread_c) < MIN_WATER_STARTS
    ):
        return None, "too_few_water_starts"
    run_lengths = [float(q) for q in np.quantile(lengths, RUN_QUANTILES)]
    gap_lengths = [float(q) for q in np.quantile(gaps, GAP_QUANTILES)]
    typical_run = float(np.median(lengths))
    typical_gap = float(np.median(gaps))

    members: list[_Member] = []
    if running:
        age = (stamp - cycles.running_since) / MINUTE  # type: ignore[operator]
        left_by_water = _minutes_left_by_water(cycles, rows, mode_now, age)
        for length in run_lengths:
            left = max(length - age, 0.5)
            if left_by_water is not None:
                left = 0.5 * left + 0.5 * left_by_water
            for gap in gap_lengths:
                rank = (len(members) * _RANK_STEP) % 1.0
                members.append(_Member(0.0, left, gap, typical_run, rank))
    else:
        errors = [float(q) for q in np.quantile(cycles.spread_c, TRIGGER_QUANTILES)]
        for error in errors:
            for length in run_lengths:
                rank = (len(members) * _RANK_STEP) % 1.0
                members.append(_Member(error, length, typical_gap, typical_run, rank))

    horizon_end = step_starts[-1] + timedelta(minutes=_STEP_MINUTES)
    minutes = max(int((horizon_end - stamp) / MINUTE), 0)
    modes: list[str] = []
    current = mode_now
    for offset in range(minutes):
        at = stamp + MINUTE * (offset + 1)
        for start, planned in zip(step_starts, planned_modes, strict=True):
            if start <= at < start + timedelta(minutes=_STEP_MINUTES):
                current = planned if planned is not None else current
                break
        modes.append(current)

    swing = max(run_w - idle_w, 1.0)
    on_minutes_needed = _STEP_MINUTES * max(threshold_w - idle_w, 0.0) / swing
    hits = np.zeros(len(step_starts))
    return_c = float(last[RETURN_COLUMN])
    triggers = {mode: cycles.trigger_c(mode) for mode in set(modes)}
    # Minutes already measured inside a step count as measured.
    measured = [
        _measured_on_minutes(rows, start, threshold_w) if start <= stamp else 0.0
        for start in step_starts
    ]
    for member in members:
        played = _play(
            cycles,
            member,
            return_c=return_c,
            running=running,
            mode_now=mode_now,
            modes=modes,
            triggers=triggers,
            stop_share=stop_share,
        )
        for index, start in enumerate(step_starts):
            first = int((start - stamp) / MINUTE)
            # ``played[0]`` is the minute after the latest measured one.
            window = played[max(first - 1, 0) : max(first - 1 + _STEP_MINUTES, 0)]
            if float(window.sum()) + measured[index] >= on_minutes_needed:
                hits[index] += 1.0
    return [float(value) for value in hits / len(members)], None


def _measured_on_minutes(
    rows: pd.DataFrame, start: datetime, threshold_w: float
) -> float:
    stamp = pd.Timestamp(start)
    window = rows[
        (rows.index >= stamp) & (rows.index < stamp + pd.Timedelta(minutes=5))
    ]
    return float((window[POWER_COLUMN] >= threshold_w).sum())


def _minutes_left_by_water(
    cycles: WaterCycles, rows: pd.DataFrame, mode: str, age: float
) -> float | None:
    """Minutes until the supply water reaches where runs in this mode end."""
    if age < RISE_AFTER_MINUTES:
        return None
    ends = [run[2] for run in cycles.runs if run[0] == mode][-STOP_TEMPERATURES:]
    if len(ends) < MIN_STOP_TEMPERATURES:
        return None
    supply = rows[SUPPLY_COLUMN].dropna()
    if supply.empty:
        return None
    earlier = supply[supply.index <= supply.index[-1] - RISE_WINDOW]
    if earlier.empty:
        return None
    rise = (float(supply.iloc[-1]) - float(earlier.iloc[-1])) / (RISE_WINDOW / MINUTE)
    if rise < MIN_RISE_PER_MIN:
        return None
    return max((float(np.median(ends)) - float(supply.iloc[-1])) / rise, 0.5)


# New initial-state wiring starts here. The numerical R1c functions above remain
# unchanged; the independently committed parity fixtures guard that boundary.
@dataclass(frozen=True)
class ControllerFit:
    """First-spell controller dynamics; enablement requires its separate CAN gate."""

    cooling: tuple[float, float, float]
    rising: tuple[float, float, float]
    cooling_sd: float
    rising_sd: float
    threshold_modes: tuple[str, ...]
    threshold_beta: tuple[float, ...]
    hysteresis: float


@dataclass(frozen=True)
class InitialState:
    """Per-path current phase and conditional remaining duration."""

    running: NDArray[np.bool_]
    elapsed_min: NDArray[np.float64]
    remaining_min: NDArray[np.float64]
    lockout_min: float
    water_draw: NDArray[np.bool_]
    tier: InputTier
    stationary: bool


def fit_controller_state(
    frame: pd.DataFrame, *, minimum_days: float, minimum_density: float
) -> ControllerFit | None:
    """Fit the controller's first-spell rates on contiguous five-minute changes.

    Args:
        frame: Qualified minute state/control/outdoor observations.
        minimum_days: Resolved owner minimum history; CAN also requires 14 days.
        minimum_density: Resolved owner minimum observed density.

    Returns:
        Dynamics or None when optional controller evidence is insufficient.
    """
    required = [
        "running",
        "control_c",
        "control_setpoint_c",
        "control_hysteresis_c",
        "outdoor_c",
        "sun",
    ]
    if frame.empty or not set(required) <= set(frame):
        return None
    valid = frame[required].notna().all(axis=1)
    rows = frame.loc[valid]
    known_index = pd.DatetimeIndex(rows.index)
    if (
        len(rows) < 2
        or (known_index[-1] - known_index[0]).total_seconds() / 86400
        < max(14.0, minimum_days)
        or valid.mean() < minimum_density
    ):
        return None
    contiguous = valid.rolling(6, min_periods=6).sum().eq(6)
    change = (frame.control_c - frame.control_c.shift(5)) / 5
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame.control_c.shift(5) - frame.outdoor_c.shift(5),
            frame.sun.shift(5),
        ]
    )
    fits: list[tuple[tuple[float, float, float], float]] = []
    for on in (False, True):
        state = frame.running.rolling(6, min_periods=6).sum().eq(6 if on else 0)
        use = contiguous & state & change.notna()
        if use.sum() < 4:
            return None
        x, y = design[use.to_numpy()], change.loc[use].to_numpy()
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ beta
        fits.append(
            (
                (float(beta[0]), float(beta[1]), float(beta[2])),
                float(np.std(residual, ddof=1)),
            )
        )
    if "mode" in rows:
        modes = tuple(sorted(str(x) for x in rows["mode"].unique()))
    else:
        modes = ()
    blocks = [np.ones(len(rows)), rows.outdoor_c.to_numpy()]
    blocks.extend(rows["mode"].eq(m).to_numpy(dtype=float) for m in modes[1:])
    beta = np.linalg.lstsq(
        np.column_stack(blocks), rows.control_setpoint_c.to_numpy(), rcond=None
    )[0]
    return ControllerFit(
        fits[0][0],
        fits[1][0],
        fits[0][1],
        fits[1][1],
        modes,
        tuple(float(v) for v in beta),
        float(rows.control_hysteresis_c.median()),
    )


def water_start_samples(
    cycles: WaterCycles,
    latest: pd.DataFrame,
    *,
    mode_now: str,
    future_modes: Sequence[str],
    origin: datetime,
) -> tuple[np.ndarray | None, str | None]:
    """Expose the legacy ensemble's distribution of its first future start.

    Args:
        cycles: Unchanged R1c fitted water parameters.
        latest: Qualified recent water/power rows.
        mode_now: Current command.
        future_modes: Known per-minute future modes, starting at the origin minute.
        origin: Actual forecast origin.

    Returns:
        First-start minutes after origin per trigger member, or a legacy guard
        reason. This is a first-start CDF, not a reduction of run probabilities.
    """
    rows = latest.dropna(subset=[POWER_COLUMN, RETURN_COLUMN])
    rows = rows.loc[rows.index <= pd.Timestamp(origin)]
    if rows.empty or origin - rows.index[-1].to_pydatetime() > MAX_MINUTE_AGE:
        return None, "stale_water"
    if not future_modes:
        return None, "no_water_horizon"
    if mode_now == _OFF and future_modes[0] != _OFF:
        return None, "unexplained_off"
    lengths = [run[1] for run in cycles.runs if run[0] == mode_now][-RECENT_RUNS:]
    if len(lengths) < MIN_RECENT_RUNS:
        lengths = [run[1] for run in cycles.runs][-RECENT_RUNS:]
    if (
        len(lengths) < MIN_RECENT_RUNS
        or len(cycles.gaps_min) < MIN_RECENT_GAPS
        or len(cycles.spread_c) < MIN_WATER_STARTS
    ):
        return None, "too_few_water_starts"
    stamp = rows.index[-1].to_pydatetime()
    lag = (origin - (stamp + MINUTE)) / MINUTE
    prefix = max(int(np.ceil(lag)), 0)
    modes = [mode_now] * prefix + list(future_modes)
    triggers = {mode: cycles.trigger_c(mode) for mode in set(modes)}
    errors = np.quantile(cycles.spread_c, TRIGGER_QUANTILES)
    run_length = float(np.median(lengths))
    gap = float(np.median(cycles.gaps_min[-RECENT_GAPS:]))
    starts = []
    for error in errors:
        played = _play(
            cycles,
            _Member(float(error), run_length, gap, run_length, 0.0),
            return_c=float(rows[RETURN_COLUMN].iloc[-1]),
            running=False,
            mode_now=mode_now,
            modes=modes,
            triggers=triggers,
            stop_share={},
        )
        first = np.flatnonzero(played)
        starts.append(max(float(first[0]) - lag, 0.0) if len(first) else float("inf"))
    return np.asarray(starts), None


def _controller_remaining(
    model: ControllerFit,
    latest: pd.DataFrame,
    running: NDArray[np.bool_],
    mode: str,
    outdoor_c: float,
    sun: float,
    rng: np.random.Generator,
) -> NDArray[np.float64] | None:
    """Solve the fitted first-spell linear ODE at current covariates."""
    if model.threshold_modes and mode not in model.threshold_modes:
        return None
    needed = ["control_c", "control_setpoint_c", "control_hysteresis_c"]
    rows = (
        latest.dropna(subset=needed) if set(needed) <= set(latest) else pd.DataFrame()
    )
    if rows.empty:
        return None
    last = rows.iloc[-1]
    beta = model.threshold_beta
    threshold = (
        beta[0]
        + beta[1] * outdoor_c
        + sum(
            beta[2 + i] * (mode == m) for i, m in enumerate(model.threshold_modes[1:])
        )
    )
    c = float(last.control_c)
    result = np.full(len(running), np.inf)
    for on, coeff, sd in [
        (False, model.cooling, model.cooling_sd),
        (True, model.rising, model.rising_sd),
    ]:
        use = running == on
        count = int(use.sum())
        if not count:
            continue
        target = threshold if on else threshold - model.hysteresis
        if c >= target if on else c <= target:
            result[use] = 0.0
            continue
        k0, k1, k2 = coeff
        drift = k0 + k2 * sun + rng.normal(0, sd, count)
        if abs(k1) < 1e-10:
            with np.errstate(divide="ignore", invalid="ignore"):
                time = (target - c) / drift
        else:
            equilibrium = outdoor_c - drift / k1
            with np.errstate(divide="ignore", invalid="ignore"):
                time = np.log((target - equilibrium) / (c - equilibrium)) / k1
        result[use] = np.where(np.isfinite(time) & (time >= 0), time, np.inf)
    return result


def initial_distribution(
    latest: pd.DataFrame,
    parameter_sets: Sequence[CycleParameters],
    cfg: HvacModelConfig,
    *,
    origin: datetime,
    mode: str,
    outdoor_c: float,
    sun: float,
    rng: np.random.Generator,
    water: WaterCycles | None = None,
    future_modes: Sequence[str] = (),
    controller: ControllerFit | None = None,
    uncertain_off: bool = False,
    off_elapsed_min: float = 0.0,
) -> InitialState:
    """Choose fresh measured phase or stationary age, then the highest valid tier.

    Args:
        latest: Qualified, classified recent minutes (availability already sliced).
        parameter_sets: Full and day-block duration fits.
        cfg: Evidence freshness, mixture and path configuration.
        origin: Actual prediction origin.
        mode: Current caller-declared mode.
        outdoor_c: Selected site outdoor at the origin.
        sun: Current radiation /1000.
        rng: Deterministic prediction generator.
        water: Optional unchanged R1c fit.
        future_modes: Known future modes for the water first-start distribution.
        controller: Optional separately qualified CAN dynamics.
        uncertain_off: Disable deterministic water schedules for uncertain OFF release.
        off_elapsed_min: Known active OFF age, if supplied by the caller.

    Returns:
        Per-path phase/age/remaining duration and the tier actually used.
    """
    n = cfg.n_paths
    selected = np.arange(n) % len(parameter_sets)
    mi = np.empty(n)
    mr = np.empty(n)
    si = np.empty(n)
    sr = np.empty(n)
    local = pd.Timestamp(origin).tz_convert(cfg.timezone)
    angle = 2 * np.pi * (local.hour + local.minute / 60) / 24
    for j, p in enumerate(parameter_sets):
        use = selected == j
        ii, rr, _ = p.locations(outdoor_c, mode, sun, np.sin(angle), np.cos(angle))
        mi[use] = ii
        mr[use] = rr
        si[use] = p.sigma_idle * cfg.dispersion_inflation
        sr[use] = p.sigma_run * cfg.dispersion_inflation
    rows = latest.dropna(subset=[POWER_COLUMN, "running"])
    fresh = bool(
        len(rows)
        and origin - rows.index[-1].to_pydatetime()
        <= timedelta(minutes=cfg.state_max_age_min)
    )
    elapsed = np.zeros(n)
    if mode in cfg.off_modes:
        running = np.zeros(n, dtype=bool)
        elapsed[:] = off_elapsed_min
        remaining = conditional_lognormal(mi, si, elapsed, rng)
    elif fresh:
        on = bool(rows.running.iloc[-1] > 0)
        running = np.full(n, on)
        # Only a contiguous suffix identifies an observed spell's elapsed time.
        i = len(rows) - 1
        while (
            i > 0
            and bool(rows.running.iloc[i - 1] > 0) == on
            and rows.index[i] - rows.index[i - 1]
            <= timedelta(minutes=cfg.minute_gap_tolerance_min)
        ):
            i -= 1
        elapsed[:] = max((origin - rows.index[i].to_pydatetime()) / MINUTE, 0.0)
        remaining = conditional_lognormal(
            np.where(running, mr, mi), np.where(running, sr, si), elapsed, rng
        )
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            mean_i = np.exp(mi + si * si / 2)
            mean_r = np.exp(mr + sr * sr / 2)
            duty = np.where(np.isposinf(mr), 1.0, mean_r / (mean_r + mean_i))
        running = rng.random(n) < duty
        mu = np.where(running, mr, mi)
        sigma = np.where(running, sr, si)
        length = np.exp(
            np.minimum(mu + sigma * sigma + sigma * rng.normal(size=n), 700)
        )
        length = np.where(np.isposinf(mu), np.inf, length)
        finite = np.isfinite(length)
        elapsed[finite] = rng.random(int(finite.sum())) * length[finite]
        remaining = np.full(n, np.inf)
        remaining[finite] = length[finite] - elapsed[finite]
    tier = InputTier.T1_POWER
    water_draw = np.zeros(n, dtype=bool)
    lockout = 0.0
    if fresh and "lockout_remaining_s" in latest:
        countdown = latest.lockout_remaining_s.dropna()
        if len(countdown):
            age = max(
                (
                    origin - (countdown.index[-1].to_pydatetime() + MINUTE)
                ).total_seconds(),
                0.0,
            )
            lockout = max(float(countdown.iloc[-1]) - age, 0.0) / 60
    if fresh and cfg.enable_t3 and controller is not None:
        fields = ["control_c", "control_setpoint_c", "control_hysteresis_c"]
        controls = (
            latest.dropna(subset=fields)
            if set(fields) <= set(latest)
            else pd.DataFrame()
        )
        if len(controls) and origin - controls.index[-1].to_pydatetime() <= timedelta(
            minutes=cfg.state_max_age_min
        ):
            projected = _controller_remaining(
                controller, controls, running, mode, outdoor_c, sun, rng
            )
            if projected is not None:
                remaining = projected
                tier = InputTier.T3_CONTROLLER
    if (
        tier is not InputTier.T3_CONTROLLER
        and fresh
        and not bool(running.any())
        and water is not None
        and not uncertain_off
        and cfg.water_weight > 0
    ):
        mapped_mode = _OFF if mode in cfg.off_modes else mode
        mapped = [_OFF if m in cfg.off_modes else m for m in future_modes]
        starts, reason = water_start_samples(
            water, latest, mode_now=mapped_mode, future_modes=mapped, origin=origin
        )
        if starts is not None and reason is None:
            water_draw = rng.random(n) < cfg.water_weight
            remaining[water_draw] = rng.choice(starts, int(water_draw.sum()))
            tier = InputTier.T2_WATER
    # The simulator applies manufacturer minima using the known elapsed age.
    remaining = np.maximum(remaining, 0.0)
    remaining[~running] = np.maximum(remaining[~running], lockout)
    return InitialState(
        running,
        elapsed,
        remaining,
        lockout,
        water_draw,
        tier,
        not fresh and mode not in cfg.off_modes,
    )
