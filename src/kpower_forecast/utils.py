import datetime
from typing import List, Union

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
        # Pysolar expects datetime with timezone info
        if t.tzinfo is None:
            # Assume UTC if naive, though strictly we should enforce awareness
            t = t.replace(tzinfo=datetime.timezone.utc)

        # get_altitude returns degrees
        alt = get_altitude(lat, lon, t)
        elevations.append(alt)

    return np.array(elevations)


def get_clear_sky_ghi(
    lat: float, lon: float, times: pd.DatetimeIndex
) -> pd.Series:
    """
    Calculate Theoretical Clear Sky GHI (Global Horizontal Irradiance)
    using pvlib.

    Args:
        lat: Latitude.
        lon: Longitude.
        times: Pandas DatetimeIndex (must be timezone aware).

    Returns:
        Pandas Series of GHI values.
    """
    location = Location(lat, lon)
    # get_clearsky returns GHI, DNI, DHI. We only need GHI.
    # Ineichen is the default model.
    clearsky = location.get_clearsky(times)
    return clearsky["ghi"]