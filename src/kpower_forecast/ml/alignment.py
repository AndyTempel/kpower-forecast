"""Timestamp-grid validation for ML forecast inference."""

from datetime import datetime

import pandas as pd

FORECAST_CONTRACT_VERSION = 2


class ForecastAlignmentError(ValueError):
    """Raised when forecast inputs or outputs do not share one interval grid."""


def prediction_origin(value: datetime, interval_minutes: int) -> pd.Timestamp:
    """Validate and normalize an explicit prediction origin.

    Args:
        value: Timezone-aware first timestamp requested by the caller.
        interval_minutes: Forecast interval size in minutes.

    Returns:
        UTC pandas timestamp aligned to the configured interval.

    Raises:
        ForecastAlignmentError: If ``value`` is naive or off-grid.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ForecastAlignmentError("prediction origin must be timezone-aware")
    origin = pd.Timestamp(value).tz_convert("UTC")
    if origin != origin.floor(f"{interval_minutes}min"):
        raise ForecastAlignmentError(
            f"prediction origin {origin.isoformat()} is not aligned to "
            f"{interval_minutes} minutes"
        )
    return origin


def validate_timestamp_grid(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    expected_start: pd.Timestamp,
    expected_length: int,
    label: str,
) -> pd.DatetimeIndex:
    """Validate a dataframe against an exact contiguous UTC timestamp grid.

    Args:
        frame: Dataframe containing a ``ds`` timestamp column.
        interval_minutes: Expected interval size in minutes.
        expected_start: Required first timestamp.
        expected_length: Required number of rows.
        label: Human-readable input name for error messages.

    Returns:
        Normalized UTC timestamp index.

    Raises:
        ForecastAlignmentError: If timestamps are missing, duplicated, or shifted.
    """
    if "ds" not in frame.columns:
        raise ForecastAlignmentError(f"{label} is missing the 'ds' timestamp column")
    if len(frame) != expected_length:
        raise ForecastAlignmentError(
            f"{label} has {len(frame)} rows; expected {expected_length}"
        )
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["ds"], utc=True))
    if timestamps.has_duplicates:
        raise ForecastAlignmentError(f"{label} contains duplicate timestamps")
    expected = pd.date_range(
        start=expected_start,
        periods=expected_length,
        freq=f"{interval_minutes}min",
        tz="UTC",
    )
    if not timestamps.equals(expected):
        actual_start = timestamps[0].isoformat() if len(timestamps) else "empty"
        actual_end = timestamps[-1].isoformat() if len(timestamps) else "empty"
        raise ForecastAlignmentError(
            f"{label} does not match the required forecast grid: "
            f"expected {expected[0].isoformat()}..{expected[-1].isoformat()}, "
            f"got {actual_start}..{actual_end}"
        )
    return timestamps
