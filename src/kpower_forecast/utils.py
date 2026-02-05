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

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        times: List of datetime objects or pandas DatetimeIndex.
               Must be timezone-aware or UTC.

    Returns:
        Numpy array of elevation angles in degrees.
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

    # Conversion factors to base units (kW or kWh)
    factors = {
        "W": 0.001,
        "kW": 1.0,
        "Wh": 0.001,
        "kWh": 1.0,
    }

    if from_unit not in factors or to_unit not in factors:
        raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")

    # Convert to kW/kWh first
    df["y"] = df["y"] * factors[from_unit]
    # Convert to target unit
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
    Handles 'cumulative_energy' and 'power' by calculating derivatives/integrals.

    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (value).
        category: 'instant_energy', 'cumulative_energy', or 'power'.
        unit: 'W', 'kW', 'Wh', or 'kWh'.
        target_interval_min: The target resolution for Prophet.

    Returns:
        pd.DataFrame: Normalized DataFrame with 'ds' and 'y' (instant kWh).
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds")

    # 1. Convert units to kW/kWh base
    # For power it's kW, for energy it's kWh
    df = convert_units(df, from_unit=unit, to_unit="kWh")

    # 2. Handle different categories
    if category == "cumulative_energy":
        # Energy between measurements
        df["y"] = df["y"].diff()
        df = df.dropna(subset=["y"])

        # Filter massive outliers (e.g. meter resets or glitches)
        # We use a simple but robust check: values > 10x the 99th percentile
        # or simply absurdly high values for a single interval.
        # For a home, > 100kWh in one interval is almost certainly a glitch.
        if not df.empty:
            q99 = df["y"].quantile(0.99)
            # If q99 is 0 (all zeros), we don't want to filter everything.
            limit = max(q99 * 10, 100.0)
            mask = df["y"] > limit
            outliers = mask.sum()
            if outliers > 0:
                import logging

                logging.getLogger(__name__).warning(
                    f"Filtered {outliers} cumulative energy outliers > {limit} kWh"
                )
                df.loc[mask, "y"] = 0  # Set to 0 to avoid breaking resampling sums

        df.loc[df["y"] < 0, "y"] = 0

    elif category == "power":
        # Average power between measurements * time delta
        # Energy(i) = (Power(i-1) + Power(i)) / 2 * (Time(i) - Time(i-1))
        # This is more accurate for power sensors.
        deltas = df["ds"].diff().dt.total_seconds() / 3600.0  # hours
        avg_power = (df["y"] + df["y"].shift(1)) / 2.0
        df["y"] = avg_power * deltas
        df = df.dropna(subset=["y"])

    # 3. Resample to target interval
    df = df.set_index("ds")
    df = df.resample(f"{target_interval_min}min").sum()
    df = df.reset_index()

    return df
