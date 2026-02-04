import datetime
from typing import List, Union

import numpy as np
import pandas as pd
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
