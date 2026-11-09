"""Extract observed sojourns without inventing transitions across missing targets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np
import pandas as pd

from .types import HvacError, HvacModelConfig, SojournEnd


@dataclass(frozen=True)
class CycleExtraction:
    """Observed spells, mode-switch responses and explicitly dropped left edges."""

    sojourns: pd.DataFrame
    mode_switches: pd.DataFrame
    dropped_left_edges: int


@dataclass
class _Spell:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    positions: list[int]
    known_start: bool
    boundary: SojournEnd | None
    interruptions: int = 0
    interruption_minutes: float = 0.0


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
        One categorical mode per timestamp; no future command is used early.
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


def extract_cycles(
    frame: pd.DataFrame,
    mode_events: pd.DataFrame,
    cfg: HvacModelConfig,
    *,
    control_events: pd.DataFrame | None = None,
) -> CycleExtraction:
    """Extract complete and right-censored spells, retaining command conditions.

    Args:
        frame: Qualified, outdoor-enriched minute frame with nullable running state.
        mode_events: All known UTC mode commands, not just changes in target power.
        cfg: Local gap, defrost and command-match durations.
        control_events: Non-mode interventions, including experimental offsets.

    Returns:
        Sojourn table and observed mode-switch responses. Spells starting at an
        unknown left edge are dropped, never called complete or right-censored.

    Raises:
        ValueError: For missing required state/target or malformed command identity.
    """
    if not {"power_w", "running"} <= set(frame):
        raise HvacError("cycle extraction requires qualified power and running state")
    events = _events(mode_events)
    modes = modes_at(pd.DatetimeIndex(frame.index), events, cfg.reference_mode)
    valid = frame.running.notna() & frame.power_w.notna()
    positions = np.flatnonzero(valid.to_numpy())
    columns = [
        "kind",
        "start",
        "end",
        "minutes",
        "end_cause",
        "censored",
        "mode",
        "mode_changed",
        "outdoor_c",
        "sun",
        "hour_sin",
        "hour_cos",
        "defrost_interruptions",
        "defrost_minutes",
        "power_w",
        "supply_c",
        "control_setpoint_c",
        "first_after_off",
    ]
    switch_columns = [
        "time",
        "from_mode",
        "to_mode",
        "state",
        "elapsed_min",
        "outcome_min",
        "outcome_censored",
    ]
    if not len(positions):
        return CycleExtraction(
            pd.DataFrame(columns=columns), pd.DataFrame(columns=switch_columns), 0
        )
    observed = frame.iloc[positions]
    observed_modes = modes.iloc[positions].to_numpy()
    kinds = np.where(
        np.isin(observed_modes, list(cfg.off_modes)),
        "off",
        np.where(observed.running.to_numpy() > 0, "run", "idle"),
    )
    stamps = observed.index
    gaps = np.r_[
        True,
        (stamps[1:] - stamps[:-1]) > pd.Timedelta(minutes=cfg.minute_gap_tolerance_min),
    ]
    changes = np.r_[True, kinds[1:] != kinds[:-1]] | gaps
    edges = np.r_[np.flatnonzero(changes), len(observed)]
    spells = []
    for a, b in pairwise(edges):
        boundary = (
            SojournEnd.GAP
            if b < len(observed) and gaps[b]
            else SojournEnd.WINDOW
            if b == len(observed)
            else None
        )
        end = stamps[b - 1] + pd.Timedelta(minutes=1) if boundary else stamps[b]
        spells.append(
            _Spell(
                str(kinds[a]),
                stamps[a],
                end,
                positions[a:b].tolist(),
                not bool(gaps[a]),
                boundary,
            )
        )
    # Merge only a short observed idle, never OFF, across no command boundary.
    merged = []
    i = 0
    while i < len(spells):
        spell = spells[i]
        while i + 2 < len(spells) and spell.kind == "run" and spell.boundary is None:
            idle, nxt = spells[i + 1], spells[i + 2]
            gap = (idle.end - idle.start).total_seconds() / 60
            if (
                idle.kind != "idle"
                or nxt.kind != "run"
                or idle.boundary is not None
                or gap >= cfg.defrost_merge_min
            ):
                break
            if ((events.time >= idle.start) & (events.time <= idle.end)).any():
                break
            gap_rows = frame.loc[(frame.index >= idle.start) & (frame.index < idle.end)]
            if (
                "defrost" in gap_rows
                and gap_rows.defrost.notna().any()
                and not gap_rows.defrost.eq(1).any()
            ):
                break
            spell = _Spell(
                "run",
                spell.start,
                nxt.end,
                spell.positions + nxt.positions,
                spell.known_start,
                nxt.boundary,
                spell.interruptions + 1 + nxt.interruptions,
                spell.interruption_minutes + gap + nxt.interruption_minutes,
            )
            i += 2
        merged.append(spell)
        i += 1
    command_times = list(events.time)
    if control_events is not None and len(control_events):
        if "time" not in control_events:
            raise HvacError("control interventions require UTC event times")
        intervention_times = pd.DatetimeIndex(control_events.time)
        if str(intervention_times.tz) != "UTC":
            raise HvacError("control intervention timestamps require UTC")
        command_times.extend(intervention_times)
    rows = []
    switches = []
    dropped = 0
    waiting_after_off = False
    for spell in merged:
        if spell.kind == "off":
            waiting_after_off = True
        if not spell.known_start:
            dropped += 1
            if spell.kind == "run":
                waiting_after_off = False
            continue
        mode = str(modes.loc[spell.start])
        next_mode = modes_at(
            pd.DatetimeIndex([spell.end]), events, cfg.reference_mode
        ).iloc[0]
        cause = spell.boundary
        if cause is None:
            cause = (
                SojournEnd.OFF
                if next_mode in cfg.off_modes and mode not in cfg.off_modes
                else (
                    SojournEnd.COMMAND
                    if any(
                        abs((at - spell.end).total_seconds())
                        <= cfg.command_match_min * 60
                        for at in command_times
                    )
                    else SojournEnd.NATURAL
                )
            )
        changed = events.loc[(events.time > spell.start) & (events.time < spell.end)]
        samples = frame.iloc[spell.positions]
        power_samples = (
            samples.loc[samples.index >= spell.start + pd.Timedelta(minutes=2)]
            if spell.kind == "run"
            else samples
        )

        def mean(column: str, samples: pd.DataFrame = samples) -> float:
            return (
                float(samples[column].mean())
                if column in samples and samples[column].notna().any()
                else float("nan")
            )

        flag_interruptions = 0
        flag_minutes = 0
        if "defrost" in samples:
            defrost_on = samples.defrost.eq(1)
            # Merged idle positions are excluded from samples, so this counts
            # only additional observed defrost while the compressor stayed on.
            flag_interruptions = int(
                (defrost_on & ~defrost_on.shift(fill_value=False)).sum()
            )
            flag_minutes = int(defrost_on.sum())
            if "defrost" in power_samples:
                power_samples = power_samples.loc[~power_samples.defrost.eq(1)]
        local = spell.start.tz_convert(cfg.timezone)
        angle = 2 * np.pi * (local.hour + local.minute / 60) / 24
        rows.append(
            dict(
                kind=spell.kind,
                start=spell.start,
                end=spell.end,
                minutes=(spell.end - spell.start).total_seconds() / 60,
                end_cause=cause.value,
                censored=cause is not SojournEnd.NATURAL,
                mode=mode,
                mode_changed=not changed.empty,
                outdoor_c=mean("outdoor_c"),
                sun=mean("sun"),
                hour_sin=float(np.sin(angle)),
                hour_cos=float(np.cos(angle)),
                defrost_interruptions=spell.interruptions + flag_interruptions,
                defrost_minutes=spell.interruption_minutes + flag_minutes,
                power_w=(
                    float(power_samples.power_w.median())
                    if len(power_samples)
                    else float("nan")
                ),
                supply_c=mean("supply_c"),
                control_setpoint_c=mean("control_setpoint_c"),
                first_after_off=spell.kind == "run" and waiting_after_off,
            )
        )
        response_events = changed
        if spell.kind == "idle":
            previous_mode = modes_at(
                pd.DatetimeIndex([spell.start - pd.Timedelta(1, unit="ns")]),
                events,
                cfg.reference_mode,
            ).iloc[0]
            if previous_mode in cfg.off_modes:
                boundary_events = events.loc[events.time.eq(spell.start)]
                response_events = pd.concat(
                    [boundary_events, changed], ignore_index=True
                )
        for event in response_events.itertuples():
            at = cast(pd.Timestamp, event.time)
            before = modes_at(
                pd.DatetimeIndex([at - pd.Timedelta(1, unit="ns")]),
                events,
                cfg.reference_mode,
            ).iloc[0]
            switches.append(
                dict(
                    time=at,
                    from_mode=str(before),
                    to_mode=str(event.mode),
                    state=spell.kind,
                    elapsed_min=(at - spell.start).total_seconds() / 60,
                    outcome_min=(spell.end - at).total_seconds() / 60,
                    outcome_censored=cause is not SojournEnd.NATURAL,
                )
            )
        if spell.kind == "run":
            waiting_after_off = False
    result = pd.DataFrame(rows, columns=columns)
    for c in ("start", "end"):
        result[c] = pd.to_datetime(result[c], utc=True)
    switches_frame = pd.DataFrame(switches, columns=switch_columns)
    switches_frame["time"] = pd.to_datetime(switches_frame.time, utc=True)
    return CycleExtraction(result, switches_frame, dropped)
