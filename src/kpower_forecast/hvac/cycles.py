"""Compressor spells from minute power, without inventing transitions across gaps.

A spell is a maximal run of one observed state (running or idle). Its start is
known only when the previous minute was observed in the other state. Its end is
natural only when the next minute is observed in the other state and no command
(mode change or control intervention) explains it; every other end is a right
censoring. Short in-run dips (defrost, metering blips) are merged into the run
and very short power blips are not runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .types import HvacError, HvacModelConfig, SojournEnd

SPELL_COLUMNS: tuple[str, ...] = (
    "kind",
    "start",
    "minutes",
    "end_cause",
    "censored",
    "mode",
    "outdoor_c",
    "sun",
    "prev_run_min",
)


@dataclass(frozen=True)
class CycleExtraction:
    """Observed spells and the number dropped for an unknown start."""

    sojourns: pd.DataFrame
    dropped_left_edges: int


def _events(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an already origin-sliced command table and preserve source."""
    if frame.empty:
        return pd.DataFrame(columns=["time", "mode", "source"])
    if "time" not in frame or "mode" not in frame:
        raise HvacError("mode history requires time and mode")
    result = frame.copy()
    timestamps = pd.DatetimeIndex(result.time)
    if timestamps.tz is None or str(timestamps.tz) != "UTC":
        raise HvacError("mode events require UTC timestamps")
    if (
        result["mode"].isna().any()
        or not result["mode"].map(lambda x: isinstance(x, str) and bool(x)).all()
    ):
        raise HvacError("mode events require nonempty mode names")
    result["time"] = timestamps
    if "source" not in result:
        result["source"] = "other"
    return result.sort_values("time", kind="stable").reset_index(drop=True)


def modes_at(
    index: pd.DatetimeIndex, mode_events: pd.DataFrame, reference_mode: str
) -> pd.Series:
    """Hold the latest known command, with reference only before any known mode.

    Args:
        index: Minute timestamps to label.
        mode_events: UTC mode changes, including commands before the first minute.
        reference_mode: Caller-declared mode when history has no previous command.

    Returns:
        One mode label per timestamp; no future command is used early.
    """
    events = _events(mode_events)
    if events.empty:
        return pd.Series(reference_mode, index=index, dtype=str)
    positions = (
        pd.DatetimeIndex(events.time)
        .as_unit("ns")
        .searchsorted(index.as_unit("ns"), side="right")
        - 1
    )
    values = np.full(len(index), reference_mode, dtype=object)
    valid = positions >= 0
    values[valid] = events["mode"].to_numpy()[positions[valid]]
    return pd.Series(values, index=index, dtype=str)


def _segments(values: NDArray[np.float64]) -> list[tuple[int, int, float]]:
    """Maximal equal-value runs as (first, last, value); NaN runs included."""
    if not len(values):
        return []
    nan = np.isnan(values)
    same = (values[1:] == values[:-1]) | (nan[1:] & nan[:-1])
    starts = np.r_[0, np.flatnonzero(~same) + 1]
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    return [
        (int(a), int(b), float(values[a])) for a, b in zip(starts, ends, strict=True)
    ]


def clean_running(
    running: pd.Series,
    *,
    max_dip_min: int,
    min_run_min: int,
    max_gap_min: int = 0,
) -> pd.Series:
    """Bridge short gaps in one state, merge in-run dips, then drop run blips.

    Args:
        running: Nullable 0/1 state on a regular minute grid.
        max_dip_min: Longest observed idle between two runs treated as part of a run.
        min_run_min: Shortest observed run kept as a compressor run.
        max_gap_min: Longest missing stretch bridged when the state is the same on
            both sides (cycle continuity only; power values stay missing). Without
            it, an invalid sample inside a long idle censors that idle, and
            censoring that depends on spell length biases the fit.

    Returns:
        A cleaned copy. Longer gaps and gaps between different states stay
        missing; dips/blips touching a gap are left unchanged.
    """
    values = running.to_numpy(dtype=float).copy()
    if max_gap_min > 0:
        segments = _segments(values)
        for k in range(1, len(segments) - 1):
            first, last, value = segments[k]
            before, after = segments[k - 1][2], segments[k + 1][2]
            if np.isnan(value) and last - first + 1 <= max_gap_min and before == after:
                values[first : last + 1] = before
    for target, other, limit in ((0.0, 1.0, max_dip_min), (1.0, 0.0, min_run_min - 1)):
        if limit < 1:
            continue
        segments = _segments(values)
        for k in range(1, len(segments) - 1):
            first, last, value = segments[k]
            if (
                value == target
                and last - first + 1 <= limit
                and segments[k - 1][2] == other
                and segments[k + 1][2] == other
            ):
                values[first : last + 1] = other
    return pd.Series(values, index=running.index)


def extract_cycles(
    frame: pd.DataFrame,
    mode_events: pd.DataFrame,
    cfg: HvacModelConfig,
    *,
    control_events: pd.DataFrame | None = None,
) -> CycleExtraction:
    """Extract complete and right-censored run/idle spells.

    Args:
        frame: Regular minute frame with nullable ``running``, ``outdoor_c``
            and ``sun``.
        mode_events: All known UTC mode commands up to the fit time.
        cfg: Gap and command-match tolerances, off modes and reference mode.
        control_events: Optional non-mode interventions; they also censor spells.

    Returns:
        One row per spell with a known start. ``prev_run_min`` is the length of
        the run immediately preceding an idle spell (NaN if unknown). Idle time
        in an off mode is never a spell: it is controlled, not thermal.

    Raises:
        ValueError: If running state or covariates are absent.
    """
    if not {"running", "outdoor_c", "sun"} <= set(frame):
        raise HvacError("cycle extraction requires running state and covariates")
    events = _events(mode_events)
    index = pd.DatetimeIndex(frame.index)
    modes = modes_at(index, events, cfg.reference_mode).to_numpy()
    state = frame.running.to_numpy(dtype=float)
    # Only commands that change something interrupt a cycle: a re-sent identical
    # mode does not. Mode commands mirrored in control_events are not counted twice.
    command = np.r_[False, modes[1:] != modes[:-1]]
    if control_events is not None and len(control_events):
        if "time" not in control_events:
            raise HvacError("control interventions require UTC event times")
        other = control_events
        if "metric" in other:
            other = other.loc[other.metric.ne("MODE_OF_OPERATION")]
        times = pd.DatetimeIndex(other.time)
        if len(times) and str(times.tz) != "UTC":
            raise HvacError("control intervention timestamps require UTC")
        positions = index.get_indexer(times.floor("min"))
        command[positions[positions >= 0]] = True
    off = np.isin(modes, list(cfg.off_modes))
    outdoor = frame.outdoor_c.to_numpy(dtype=float)
    sun = frame.sun.to_numpy(dtype=float)
    rows = []
    dropped = 0
    previous: tuple[float, int, int] | None = None
    match = cfg.command_match_min
    for first, last, value in _segments(state):
        if np.isnan(value):
            previous = None
            continue
        known_start = first > 0 and not np.isnan(state[first - 1])
        after = last + 1
        end_observed = after < len(state) and not np.isnan(state[after])
        length = last - first + 1
        cause = (
            SojournEnd.NATURAL
            if end_observed
            else (SojournEnd.WINDOW if after >= len(state) else SojournEnd.GAP)
        )
        # Commands are logged at or before the minute their effect shows, so
        # one in [end - match, first minute after the spell] explains the end;
        # an earlier one inside the spell truncates it (the rest has no known
        # start). Either way the observed length is only a lower bound.
        window = max(after - match, first + 1)
        hits = np.flatnonzero(command[first + 1 : min(after + 1, len(state))])
        if len(hits) and first + 1 + hits[0] < window:
            length, cause = int(hits[0]) + 1, SojournEnd.COMMAND
        elif len(hits) and cause is SojournEnd.NATURAL:
            cause = SojournEnd.COMMAND
        if (
            cause is SojournEnd.NATURAL
            and after < len(state)
            and off[after] != off[first]
        ):
            cause = SojournEnd.OFF
        prev_run = (
            float(previous[1])
            if previous is not None and previous[0] == 1.0 and previous[2] == first - 1
            else np.nan
        )
        # A spell with an unknown start has an unknown length.
        previous = (value, last - first + 1, last) if known_start else None
        if not known_start:
            dropped += 1
            continue
        if off[first] and value == 0.0:
            continue
        rows.append(
            (
                "run" if value == 1.0 else "idle",
                index[first],
                float(length),
                cause.value,
                cause is not SojournEnd.NATURAL,
                str(modes[first]),
                float(outdoor[first]),
                float(sun[first]),
                prev_run,
            )
        )
    result = pd.DataFrame(rows, columns=list(SPELL_COLUMNS))
    result["start"] = pd.to_datetime(result.start, utc=True)
    return CycleExtraction(result, dropped)
