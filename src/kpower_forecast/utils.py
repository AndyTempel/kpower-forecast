import datetime
from typing import List, Union, cast

import numpy as np
import pandas as pd
from pvlib.location import Location
from pysolar.solar import get_altitude


def calculate_solar_elevation(
    lat: float, lon: float, times: Union[pd.DatetimeIndex, List[datetime.datetime]]
) -> np.ndarray:
    """
    Calculate solar elevation angles (altitude) for a list of times
    at a specific location.
    """
    elevations = []
    for t in times:
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        alt = get_altitude(lat, lon, t)
        elevations.append(alt)
    return np.array(elevations)


def get_clear_sky_ghi(lat: float, lon: float, times: pd.DatetimeIndex) -> pd.Series:
    """
    Calculate Theoretical Clear Sky GHI (Global Horizontal Irradiance)
    using pvlib.
    """
    location = Location(lat, lon)
    clearsky = location.get_clearsky(times)
    return cast(pd.Series, clearsky["ghi"])


def convert_units(
    df: pd.DataFrame, from_unit: str, to_unit: str = "kWh"
) -> pd.DataFrame:
    """
    Converts values in 'y' column to target unit.
    Supported: W, kW, Wh, kWh.
    """
    df = df.copy()
    if from_unit == to_unit:
        return df

    factors = {
        "W": 0.001,
        "kW": 1.0,
        "Wh": 0.001,
        "kWh": 1.0,
    }

    if from_unit not in factors or to_unit not in factors:
        raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")

    df["y"] = df["y"] * factors[from_unit]
    df["y"] = df["y"] / factors[to_unit]

    return df


def normalize_to_instant_kwh(
    df: pd.DataFrame,
    category: str,
    unit: str,
    target_interval_min: int = 15,
) -> pd.DataFrame:
    """
    Normalizes input data to 'instant_energy' in 'kWh' with consistent intervals.
    Uses Index Union + Reindexing for mathematically correct cumulative interpolation.
    This prevents energy aliasing/spikes regardless of input irregularity.
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")

    # 1. Convert units to kWh base (kW for power)
    df = convert_units(df, from_unit=unit, to_unit="kWh")

    # 2. Construct Continuous Cumulative Series
    if category == "instant_energy":
        df["y"] = df["y"].fillna(0).cumsum()
    elif category == "power":
        # Integrate Power (kW) -> Energy (kWh)
        seconds = df["ds"].diff().dt.total_seconds().fillna(0)
        hours = seconds / 3600.0
        # Rectangular integration (previous power * duration)
        energy_step = df["y"].shift(1).fillna(0) * hours
        df["y"] = energy_step.cumsum()
    elif category == "cumulative_energy":
        # Ensure monotonic (handle resets by keeping relative growth)
        diffs = df["y"].diff().fillna(0)
        diffs[diffs < 0] = 0
        df["y"] = diffs.cumsum()

    # 3. Robust Interpolation using Index Union
    # Create target fixed-frequency index
    start = df["ds"].min().floor(f"{target_interval_min}min")
    end = df["ds"].max().ceil(f"{target_interval_min}min")
    target_index = pd.date_range(
        start=start, end=end, freq=f"{target_interval_min}min", tz="UTC"
    )

    # Combine indices to preserve actual measurement points for slope calculation
    df = df.set_index("ds")
    combined_index = df.index.union(target_index)

    # Reindex to combined points, interpolate, then downsample to target grid
    df_normalized = (
        df.reindex(combined_index).interpolate(method="time").reindex(target_index)
    )

    # 4. Differentiate to get Instant Energy per Interval
    df_final = df_normalized.diff().fillna(0)
    df_final = df_final.reset_index().rename(columns={"index": "ds"})

    # 5. Safety: Clip negative values
    df_final.loc[df_final["y"] < 0, "y"] = 0

    return df_final
