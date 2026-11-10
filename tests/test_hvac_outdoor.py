"""MOS provenance, conditioned uncertainty, and hourly residual persistence."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.outdoor import OutdoorFit, fit_outdoor, outdoor_paths


def test_mos_uses_forecasts_and_marks_reanalysis_fallback():
    index = pd.date_range("2026-10-01", periods=240, freq="h", tz="UTC")
    rng = np.random.default_rng(5)
    temperature = 7 + rng.normal(size=240)
    weather = pd.DataFrame(
        {
            "temperature_2m": temperature,
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 2.0,
            "weather_source": "historical_forecast",
        },
        index=index,
    )
    site = pd.DataFrame(
        {
            "outdoor_c": 2 + temperature * 0.9 + rng.normal(0, 0.1, 240),
            "outdoor_source": "environment",
        },
        index=index,
    )
    fit = fit_outdoor(site, weather, "UTC")
    assert not fit.on_reanalysis and fit.lead_unaware
    assert fit.mean(weather, "UTC") == pytest.approx(site.outdoor_c, abs=0.5)
    weather.loc[index[48:], "weather_source"] = "reanalysis"
    short = fit_outdoor(site, weather, "UTC")
    assert short.on_reanalysis
    site["outdoor_source"] = "weather"
    with pytest.raises(ValueError, match="insufficient_weather"):
        fit_outdoor(site, weather, "UTC")


def test_outdoor_conditioning_and_ar1_spread():
    origin = datetime(2026, 11, 9, tzinfo=UTC)
    weather = pd.DataFrame(
        {"temperature_2m": 7.0, "shortwave_radiation": 0.0, "wind_speed_10m": 0.0},
        index=pd.date_range(origin, periods=25, freq="h"),
    )
    index = pd.date_range(origin, periods=24 * 60, freq="min")
    model = OutdoorFit((0.0, 1.0, 0.0, 0.0, 0.0, 0.0), 0.8, 2.0, False)
    paths = outdoor_paths(
        model,
        weather,
        index,
        origin=origin,
        timezone="UTC",
        n_paths=4096,
        rng=np.random.default_rng(4),
        observed_c=9.0,
    )
    assert np.all(paths[:, 0] == 9.0)
    assert paths[:, 60].std() == pytest.approx(2 * np.sqrt(1 - 0.8**2), abs=0.06)
    assert paths[:, -1].std() == pytest.approx(2.0, abs=0.08)
    unknown = outdoor_paths(
        model,
        weather,
        index,
        origin=origin,
        timezone="UTC",
        n_paths=4096,
        rng=np.random.default_rng(4),
        observed_c=None,
    )
    assert unknown[:, 0].std() == pytest.approx(2.0, abs=0.08)
    with pytest.raises(ValueError, match="does not cover"):
        outdoor_paths(
            model,
            weather.iloc[:2],
            index,
            origin=origin,
            timezone="UTC",
            n_paths=16,
            rng=np.random.default_rng(4),
            observed_c=9.0,
        )


def test_weather_only_calibrates_against_reanalysis_not_itself():
    index = pd.date_range("2026-10-01", periods=240, freq="h", tz="UTC")
    rng = np.random.default_rng(91)
    truth = 7 + rng.normal(size=len(index))
    common = {"shortwave_radiation": 0.0, "wind_speed_10m": 2.0}
    forecast = pd.DataFrame(
        {
            **common,
            "temperature_2m": truth + rng.normal(1.0, 1.0, len(index)),
            "weather_source": "historical_forecast",
        },
        index=index,
    )
    reanalysis = pd.DataFrame(
        {**common, "temperature_2m": truth, "weather_source": "reanalysis"}, index=index
    )
    frame = pd.DataFrame(
        {"outdoor_c": forecast.temperature_2m, "outdoor_source": "weather"}, index=index
    )
    fit = fit_outdoor(frame, pd.concat([forecast, reanalysis]), "UTC")
    assert fit.weather_reference and fit.stationary_sd_c > 0.5
    assert not fit.on_reanalysis
