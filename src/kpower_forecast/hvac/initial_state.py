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
