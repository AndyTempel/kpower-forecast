"""Causal, evidence-preserving minute preparation and outdoor selection.

The history policy is resolved by the owner and supplied with the data. Nothing
here supplies alternative eligibility thresholds or reconstructs target values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd

from .types import HistoryPolicy, HvacError, HvacModelConfig, utc, validate_minute_index

SIGNALS: tuple[str, ...] = (
    "power_w",
    "running",
    "defrost",
    "lockout_remaining_s",
    "control_c",
    "control_setpoint_c",
    "control_hysteresis_c",
    "return_c",
    "supply_c",
    "outdoor_unit_c",
    "outdoor_env_c",
)


@dataclass(frozen=True)
class HistoryVerdict:
    """Owner-policy evidence, separate from cycle-model training sufficiency."""

    span_days: float | None
    observed_density: float | None
    full_row_fraction: float | None
    mean_coverage: float | None
    reliable: bool
    reason: str | None
    observed_minutes: int
    missing_minutes: int
    bridged_minutes: int


def policy_number(policy: HistoryPolicy, name: str) -> float:
    """Read a required resolved setting without inventing a default.

    Args:
        policy: Owner policy and version.
        name: Exact owning setting name.

    Returns:
        Its finite numeric value, or seconds for a duration setting.

    Raises:
        ValueError: If missing, malformed, nonfinite or contradictory.
    """
    value = policy.settings.get(name)
    if isinstance(value, dict):
        value = value.get("seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
    ):
        raise HvacError(f"missing or invalid owner history-policy setting: {name}")
    if policy.settings.get("_HISTORY_POLICY_VERSION") != policy.version:
        raise HvacError("history-policy version contradicts its settings")
    return float(value)


def history_window(frame: pd.DataFrame, policy: HistoryPolicy) -> pd.DataFrame:
    """Select rows after the latest overlong target gap, retaining all shorter gaps.

    Args:
        frame: Regular UTC minutes with qualified target values.
        policy: Resolved owning history policy.

    Returns:
        A copy of the eligible latest window; no value is filled.

    Raises:
        ValueError: For an invalid policy.
    """
    gap_s = policy_number(policy, "_MAX_TRAINING_GAP")
    if gap_s < 0:
        raise HvacError("owner maximum training gap cannot be negative")
    if frame.empty:
        return frame.copy()
    valid = frame.power_w.notna()
    group = valid.ne(valid.shift(fill_value=True)).cumsum()
    counts = group.loc[~valid].value_counts()
    oversized = counts[counts > int(gap_s // 60)].index
    if not len(oversized):
        return frame.copy()
    end = frame.index[(~valid) & group.eq(int(oversized.max()))].max()
    return cast(pd.DataFrame, frame.loc[frame.index > end].copy())


def history_verdict(
    frame: pd.DataFrame, policy: HistoryPolicy, *, as_of: datetime
) -> HistoryVerdict:
    """Evaluate the owning span, density, change-age and applicable coverage rules.

    Args:
        frame: Qualified latest history window at one-minute resolution.
        policy: Resolved settings and scope; direct meters have no attribution gate.
        as_of: Causal assessment time, never the workstation's current clock.

    Returns:
        The verdict plus measured evidence and missing/bridged counts.
    """
    as_of = utc(as_of)
    observed = int(frame.power_w.notna().sum())
    span = (
        (frame.index[-1] - frame.index[0]).total_seconds() / 86400
        if len(frame) > 1
        else None
    )
    expected = (
        int((frame.index[-1] - frame.index[0]).total_seconds() // 60) + 1
        if len(frame) > 1
        else 0
    )
    density = min(observed / expected, 1.0) if expected else None
    reason = None
    if span is None or span < policy_number(policy, "_MIN_SPAN_DAYS"):
        reason = "insufficient_history"
    elif density is None or density < policy_number(policy, "_MIN_DENSITY"):
        reason = "low_data_density"
    elif policy.last_fingerprint_change_at is not None and (
        as_of - policy.last_fingerprint_change_at
    ).total_seconds() / 3600 < policy_number(policy, "_MIN_HOURS_SINCE_CHANGE"):
        reason = "recent_device_change"
    full = mean = None
    if policy.scope == "system_attributed_heating":
        rows = frame.loc[frame.power_w.notna()]
        if len(rows):
            rows = rows.loc[
                rows.index
                >= rows.index.max()
                - pd.Timedelta(
                    seconds=policy_number(policy, "_ATTRIBUTION_QUALITY_WINDOW")
                )
            ]
        coverage = (
            pd.to_numeric(
                rows.get("coverage_ratio", pd.Series(dtype=float)), errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )
        full = (
            float(
                (
                    coverage
                    >= policy_number(policy, "_FULL_ATTRIBUTION_COVERAGE_RATIO")
                ).mean()
            )
            if len(coverage)
            else 0.0
        )
        mean = float(coverage.mean()) if len(coverage) else 0.0
        if full < policy_number(
            policy, "_MIN_FULL_ATTRIBUTION_ROW_FRACTION"
        ) or mean < policy_number(policy, "_MIN_MEAN_ATTRIBUTION_COVERAGE"):
            reason = "partial_attribution_coverage"
    bridged = int(
        frame.get("power_w__bridged", pd.Series(dtype=bool)).fillna(False).sum()
    )
    return HistoryVerdict(
        span,
        density,
        full,
        mean,
        reason is None,
        reason,
        observed,
        len(frame) - observed,
        bridged,
    )


def prepare_minutes(
    frame: pd.DataFrame,
    cfg: HvacModelConfig,
    *,
    as_of: datetime,
    policy: HistoryPolicy | None = None,
    epoch_start: datetime | None = None,
    training: bool = True,
) -> pd.DataFrame:
    """Slice availability first, qualify observations, then regularize without filling.

    Args:
        frame: Values and per-signal evidence on UTC minute-start labels.
        cfg: Gap, ingestion and training-window configuration.
        as_of: Fit/forecast origin.
        policy: Required for selecting a training-history window.
        epoch_start: Optional lineage boundary.
        training: Apply training-days and owner gap-window rules when True.

    Returns:
        Regular minute grid with invalid/unknown or bridged observations as NaN;
        evidence columns remain intact for diagnostics.

    Raises:
        ValueError: For invalid identity or missing training policy.
    """
    validate_minute_index(frame)
    cutoff = utc(as_of) - timedelta(minutes=1 + cfg.ingest_lag_min)
    lower = utc(epoch_start) if epoch_start is not None else None
    if training:
        lower = (
            max(lower, utc(as_of) - timedelta(days=cfg.training_days))
            if lower
            else utc(as_of) - timedelta(days=cfg.training_days)
        )
        if policy is None:
            raise HvacError("training requires the resolved owner history policy")
    selected = frame.loc[
        (frame.index <= cutoff) & (frame.index >= lower if lower else True)
    ].copy()
    if selected.empty:
        return selected
    selected = selected.reindex(
        pd.date_range(selected.index[0], selected.index[-1], freq="min", name="time")
    )
    for column in SIGNALS:
        if column not in selected:
            continue
        values = pd.to_numeric(selected[column], errors="raise")
        if np.isinf(values.dropna().to_numpy(dtype=float)).any():
            raise HvacError(f"nonfinite {column}")
        # Only explicit invalidity rejects an observed value. Aggregated minutes
        # older than the raw-retention window carry no per-sample quality tag;
        # treating "unknown" as invalid discarded most of the history (the EMS
        # forecast models use those rows too). Missing values stay missing.
        flag = selected.get(
            column + "__measured_valid", pd.Series(pd.NA, index=selected.index)
        )
        mask = ~flag.eq(False).fillna(False).astype(bool)
        if column + "__quality" in selected and policy:
            quality = selected[column + "__quality"]
            accepted = set(policy.measured_valid_qualities) | {"unknown"}
            mask &= quality.isna() | quality.isin(accepted)
        bridged = (
            selected.get(column + "__bridged", pd.Series(False, index=selected.index))
            .fillna(False)
            .eq(True)
        )
        if column in {"outdoor_env_c", "outdoor_unit_c"}:
            invalid = selected.get(
                column + "__measured_valid",
                pd.Series(index=selected.index, dtype=object),
            ).eq(False)
            name = (
                "outdoor_env_invalid"
                if column == "outdoor_env_c"
                else "outdoor_unit_invalid"
            )
            previous = (
                selected.get(name, pd.Series(0.0, index=selected.index)).fillna(0).eq(1)
            )
            selected[name] = (previous | invalid).astype(float)
        selected[column] = values.where(mask & ~bridged)
        if (
            column in {"running", "defrost"}
            and not selected[column].dropna().isin([0, 1]).all()
        ):
            raise HvacError(f"{column} must be a qualified binary flag")
    return (
        history_window(selected, cast(HistoryPolicy, policy)) if training else selected
    )


def weather_minutes(weather: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate available hourly weather, never extrapolating its edges.

    Args:
        weather: UTC hourly index or a long source-labelled table with `ds`.
        index: Requested UTC minute labels.

    Returns:
        Numeric weather columns on the requested index, with source priority
        historical_forecast then reanalysis where both are present.
    """
    columns = [
        "temperature_2m",
        "shortwave_radiation",
        "wind_speed_10m",
        "relative_humidity_2m",
    ]
    if weather.empty:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    source = weather.copy()
    if "ds" in source:
        source["ds"] = pd.to_datetime(source.ds, utc=True)
        if "weather_source" in source:
            source["_priority"] = source.weather_source.map(
                {"historical_forecast": 0, "reanalysis": 1}
            ).fillna(2)
            source = source.sort_values(["ds", "_priority"])
        source = source.groupby("ds", sort=True)[
            [c for c in columns if c in source]
        ].first()
    if (
        not isinstance(source.index, pd.DatetimeIndex)
        or str(source.index.tz) != "UTC"
        or not source.index.is_unique
    ):
        raise HvacError("weather requires unique UTC hourly identity")
    numeric = source.reindex(columns=columns).apply(pd.to_numeric, errors="raise")
    return cast(
        pd.DataFrame,
        (
            numeric.reindex(numeric.index.union(index))
            .sort_index()
            .interpolate(method="time", limit_area="inside")
            .reindex(index)
        ),
    )


def _held_sensor(
    values: pd.Series,
    invalid: pd.DatetimeIndex,
    index: pd.DatetimeIndex,
    hold_min: int,
    grace_min: int,
) -> pd.Series:
    """Vectorize the exact R1c read-age/first-invalidation semantics."""
    known = values.dropna().sort_index()
    output = np.full(len(index), np.nan)
    if known.empty:
        return pd.Series(output, index=index)
    known_index = pd.DatetimeIndex(known.index).as_unit("ns")
    positions = known_index.searchsorted(index.as_unit("ns"), side="right") - 1
    safe = np.maximum(positions, 0)
    read = pd.DatetimeIndex(known_index[safe])
    age = (index - read).total_seconds() / 60
    first = invalid.as_unit("ns").searchsorted(read.as_unit("ns"), side="right")
    elapsed = np.full(len(index), -np.inf)
    matched = first < len(invalid)
    elapsed[matched] = (
        np.asarray(index.as_unit("ns").astype("int64"))[matched]
        - np.asarray(invalid.as_unit("ns").astype("int64"))[first[matched]]
    ) / 60e9
    valid = (
        (positions >= 0)
        & (age <= hold_min)
        & ~((first < len(invalid)) & (elapsed > grace_min))
    )
    output[valid] = known.to_numpy(dtype=float)[safe[valid]]
    return pd.Series(output, index=index)


def outdoor_frame(
    frame: pd.DataFrame, weather: pd.DataFrame, cfg: HvacModelConfig
) -> pd.DataFrame:
    """Choose unit, environmental or weather values in that order at each minute.

    Args:
        frame: Qualified minute signals and explicit invalidation markers.
        weather: Available hourly weather (future vintages selected by the caller).
        cfg: Hold and environmental invalidation-grace durations.

    Returns:
        A copied frame with outdoor_c, outdoor_source and sun covariates.
    """
    result = frame.copy()
    weather_grid = weather_minutes(weather, pd.DatetimeIndex(result.index))
    result["outdoor_c"] = np.nan
    result["outdoor_source"] = "unknown"
    for column, marker, grace, source in [
        ("outdoor_unit_c", "outdoor_unit_invalid", 0, "unit"),
        (
            "outdoor_env_c",
            "outdoor_env_invalid",
            cfg.outdoor_invalid_grace_min,
            "environment",
        ),
    ]:
        values = result.get(column, pd.Series(dtype=float, index=result.index))
        invalid = pd.DatetimeIndex(
            result.index[
                result.get(marker, pd.Series(0.0, index=result.index)).fillna(0).eq(1)
            ]
        )
        held = _held_sensor(
            values, invalid, pd.DatetimeIndex(result.index), cfg.outdoor_hold_min, grace
        )
        use = result.outdoor_c.isna() & held.notna()
        result.loc[use, "outdoor_c"] = held[use]
        result.loc[use, "outdoor_source"] = source
    use = result.outdoor_c.isna() & weather_grid.temperature_2m.notna()
    result.loc[use, "outdoor_c"] = weather_grid.temperature_2m[use]
    result.loc[use, "outdoor_source"] = "weather"
    for c in ("shortwave_radiation", "wind_speed_10m", "relative_humidity_2m"):
        result[c] = weather_grid[c]
    result["sun"] = result.shortwave_radiation / 1000
    return result


@dataclass(frozen=True)
class RunningClassification:
    """A learned guard with an explicit not-cycling outcome."""

    threshold_w: float | None
    running: pd.Series
    reason: str | None


def running_states(
    frame: pd.DataFrame, threshold_w: float | None = None
) -> RunningClassification:
    """Learn a log-power histogram valley or use a caller-declared running guard.

    Args:
        frame: Qualified power and optional measured compressor flags.
        threshold_w: Optional declared guard in watts.

    Returns:
        Nullable state, threshold and an explicit noncycling reason when needed.
    """
    power = frame.power_w.dropna()
    if threshold_w is None:
        logs = np.log10(np.maximum(power.to_numpy(dtype=float), 1.0))
        if len(logs) < 20 or np.ptp(logs) < 0.2:
            return RunningClassification(
                None, pd.Series(np.nan, index=frame.index), "not_cycling"
            )
        counts, edges = np.histogram(logs, bins=60)
        smooth = np.convolve(counts, [1, 2, 3, 2, 1], mode="same")
        peaks = [
            i
            for i in range(60)
            if smooth[i] > 0
            and (i == 0 or smooth[i] >= smooth[i - 1])
            and (i == 59 or smooth[i] >= smooth[i + 1])
        ]
        ranked = sorted(peaks, key=lambda i: smooth[i], reverse=True)
        pair = next(((a, b) for a in ranked for b in ranked if abs(a - b) >= 4), None)
        if pair is None:
            return RunningClassification(
                None, pd.Series(np.nan, index=frame.index), "not_cycling"
            )
        low, high = sorted(pair)
        valley = low + int(np.argmin(smooth[low : high + 1]))
        threshold_w = float(10 ** ((edges[valley] + edges[valley + 1]) / 2))
        upper = power[power > threshold_w]
        if (
            len(upper) < 0.05 * len(power)
            or len(upper) < 2
            or upper.std() / upper.mean() > 0.25
        ):
            return RunningClassification(
                None, pd.Series(np.nan, index=frame.index), "not_cycling"
            )
        # The histogram valley is wide and flat between standby and full power;
        # its first minimum sits just above standby, so partial start/stop
        # minutes and pump tails would count as full-power run minutes. A
        # quarter of the typical run power separates them robustly.
        threshold_w = float(max(0.25 * upper.median(), 200.0))
    if not np.isfinite(threshold_w) or threshold_w <= 0:
        raise HvacError("running threshold must be positive finite watts")
    state = (frame.power_w > threshold_w).astype(float).where(frame.power_w.notna())
    if "running" in frame:
        state = frame.running.combine_first(state).where(frame.power_w.notna())
    return RunningClassification(threshold_w, state, None)
