"""Snow-on-panel load model."""

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.snow_cover import (
    SnowCoverParameters,
    SnowCoverSource,
    SnowLoadState,
    observed_cover,
    step_snow_load,
)

_START = pd.Timestamp("2026-01-01T00:00Z")


def _sunny_ghi(ds: pd.DatetimeIndex, peak_w_m2: float = 400.0) -> np.ndarray:
    """A clear winter day: sun from 07:00 to 16:00 UTC."""
    hour = ds.hour + ds.minute / 60.0
    return np.clip(peak_w_m2 * np.sin(np.pi * (hour - 7.0) / 9.0), 0.0, None)


@pytest.fixture
def snow_spell() -> pd.DataFrame:
    """Twelve days, hourly: 10 cm of snow on day 2, a cold sunny week, a thaw.

    The ground keeps 5 cm of snow through the thaw, so ground depth stays
    high after the panels have shed theirs.
    """
    ds = pd.date_range(_START, periods=12 * 24, freq="h")
    day = (ds - _START).days
    frame = pd.DataFrame(
        {
            "ds": ds,
            "snowfall": 0.0,
            "temperature_2m": np.where(day < 2, 4.0, -3.0),
            "shortwave_radiation": _sunny_ghi(ds),
            "snow_depth": np.where(day < 2, 0.0, 0.05),
        }
    )
    snowing = (day == 2) & (ds.hour < 10)
    frame.loc[snowing, "snowfall"] = 1.0  # cm/h
    frame.loc[snowing, "shortwave_radiation"] = 0.0
    frame.loc[day >= 9, "temperature_2m"] = 8.0
    return frame


def _daily_factor(weather: pd.DataFrame, factor: np.ndarray) -> pd.Series:
    """Irradiance-weighted cover factor per day."""
    frame = pd.DataFrame(
        {
            "day": (weather["ds"] - _START).dt.days,
            "ghi": weather["shortwave_radiation"],
            "weighted": weather["shortwave_radiation"] * factor,
        }
    )
    daylight = frame.loc[frame["ghi"] > 0].groupby("day")
    return daylight["weighted"].sum() / daylight["ghi"].sum()


def test_fresh_snow_zeroes_output_until_it_sheds(snow_spell) -> None:
    trajectory = step_snow_load(snow_spell, 60, SnowCoverParameters())
    daily = _daily_factor(snow_spell, trajectory.factor)

    assert daily[[0, 1]].min() == 1.0
    # Cold sun sheds about 2.3 cm a day, so 10 cm keep the panels covered
    # for days without any new snowfall.
    assert daily[[3, 4, 5]].max() == 0.0


def test_ground_snow_does_not_cover_shed_panels(snow_spell) -> None:
    trajectory = step_snow_load(snow_spell, 60, SnowCoverParameters())
    daily = _daily_factor(snow_spell, trajectory.factor)

    # 5 cm stays on the ground, but the thaw cleared the panels.
    thaw = (snow_spell["ds"] - _START).dt.days >= 10
    assert snow_spell.loc[thaw, "snow_depth"].min() >= 0.05
    assert daily[[10, 11]].min() == 1.0


def test_load_continues_from_a_stored_state() -> None:
    weather = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-08T07:00Z", periods=8, freq="h"),
            "snowfall": 0.0,
            "temperature_2m": -2.0,
            "shortwave_radiation": 150.0,
        }
    )
    initial = SnowLoadState(
        load_cm=5.0,
        source=SnowCoverSource.OBSERVED,
        at=pd.Timestamp("2026-01-08T06:00Z"),
    )

    trajectory = step_snow_load(weather, 60, SnowCoverParameters(), initial=initial)

    # Cold daylight sheds 0.15 cm/h: still covered without any new snowfall.
    assert trajectory.factor.max() == 0.0
    assert trajectory.final_state().source == SnowCoverSource.OBSERVED


def _observed(actual: list[float], temperature: float) -> np.ndarray:
    expected = np.full(len(actual), 1.0)  # 4 kW mean over 15 minutes
    return observed_cover(
        pd.Series(actual),
        expected,
        np.full(len(actual), temperature),
        15,
        SnowCoverParameters(),
    )


def test_observed_cover_needs_an_hour_of_cold_dark_output() -> None:
    covered = _observed([0.0] * 6, temperature=-1.0)

    # Known only once the hour is complete.
    assert covered.tolist() == [0, 0, 0, 1, 1, 1]
    # The same output when warm is an outage, not snow.
    assert not _observed([0.0] * 6, temperature=8.0).any()


def test_observed_output_clears_the_panels() -> None:
    weather = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-14T09:00Z", periods=4, freq="15min"),
            "snowfall": 0.0,
            "temperature_2m": -1.0,
            "shortwave_radiation": 300.0,
        }
    )
    observation = _observed([0.0, 0.6, 0.7, 0.8], temperature=-1.0)
    initial = SnowLoadState(
        load_cm=10.0,
        source=SnowCoverSource.WEATHER,
        at=pd.Timestamp("2026-01-14T08:45Z"),
    )

    trajectory = step_snow_load(
        weather, 15, SnowCoverParameters(), initial=initial, observation=observation
    )

    assert trajectory.factor.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert trajectory.final_state().source == SnowCoverSource.OBSERVED
