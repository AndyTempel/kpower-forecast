import numpy as np
import pandas as pd
import pytest

from kpower_forecast.ml.solar_physics import (
    fit_geometry,
    irradiance_components,
    plane_of_array,
    sun_position,
)

LATITUDE, LONGITUDE = 46.0, 14.3


def clear_sky_weather(
    start: str, days: int, *, overcast_every: int = 0
) -> pd.DataFrame:
    """15-minute weather with a clear-sky direct/diffuse split.

    Every ``overcast_every``-th day (if set), starting with the first, has
    15 % of the clear-sky global irradiance, all diffuse, at 100 % cloud cover.
    """
    from pvlib.location import Location

    ds = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    sky = Location(LATITUDE, LONGITUDE).get_clearsky(ds)
    sun = sun_position(pd.Series(ds), LATITUDE, LONGITUDE, 15)
    cos_zenith = np.clip(np.cos(np.radians(sun["zenith"].to_numpy())), 0.0, None)
    direct = sky["dni"].to_numpy() * cos_zenith
    diffuse = sky["dhi"].to_numpy()
    cloud = np.full(len(ds), 10.0)
    if overcast_every:
        dull = (np.arange(len(ds)) // 96) % overcast_every == 0
        diffuse = np.where(dull, 0.15 * (direct + diffuse), diffuse)
        direct = np.where(dull, 0.0, direct)
        cloud = np.where(dull, 100.0, cloud)
    return pd.DataFrame(
        {
            "ds": ds,
            "shortwave_radiation": direct + diffuse,
            "direct_radiation": direct,
            "diffuse_radiation": diffuse,
            "cloud_cover": cloud,
            "temperature_2m": 12.0,
            "clear_sky_ghi": sky["ghi"].to_numpy(),
        }
    )


def plane_energy(
    weather: pd.DataFrame, tilt: float, azimuth: float, lag: int = 0
) -> pd.Series:
    """Interval energy of a 10 kWp-like array on the given plane."""
    sun = sun_position(weather["ds"], LATITUDE, LONGITUDE, 15)
    components = irradiance_components(weather, sun, lag)
    return pd.Series(0.0025 * plane_of_array(components, sun, tilt, azimuth))


def test_geometry_and_radiation_lag_are_recovered_from_history() -> None:
    weather = clear_sky_weather("2026-09-01", 20)
    target = plane_energy(weather, tilt=35.0, azimuth=200.0, lag=1)
    sun = sun_position(weather["ds"], LATITUDE, LONGITUDE, 15)

    geometry = fit_geometry(target, weather, sun, sun["elevation"].to_numpy() > 8)

    assert geometry.fitted
    assert geometry.lag_steps == 1
    assert geometry.tilt_deg == pytest.approx(35.0, abs=5.0)
    assert geometry.azimuth_deg == pytest.approx(200.0, abs=10.0)


def test_without_a_direct_diffuse_split_irradiance_is_decomposed() -> None:
    weather = clear_sky_weather("2026-06-21", 1).drop(
        columns=["direct_radiation", "diffuse_radiation"]
    )
    sun = sun_position(weather["ds"], LATITUDE, LONGITUDE, 15)

    components = irradiance_components(weather, sun)

    noon = sun["elevation"].idxmax()
    assert components.loc[noon, "dni"] > 500.0
    assert components.loc[noon, "dhi"] < components.loc[noon, "ghi"]
    assert (components.loc[sun["elevation"] < 0, "dni"] == 0.0).all()
