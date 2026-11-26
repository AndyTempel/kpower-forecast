"""Snow-cover gate through ML training and prediction."""

import pandas as pd
import pytest

from kpower_forecast.ml import (
    HybridStructure,
    KPowerMLForecast,
    MLBackendType,
    MLForecastType,
)
from kpower_forecast.utils import get_clear_sky_ghi

_LAT = 46.0
_LON = 14.0
_KWH_PER_W_M2 = 10.0 * 0.25 / 1000.0  # a 10 kWp array per 15-minute row


def _site(
    tmp_path, monkeypatch, weather: pd.DataFrame, snow_free_until: pd.Timestamp
) -> KPowerMLForecast:
    forecast = KPowerMLForecast(
        model_id="solar",
        latitude=_LAT,
        longitude=_LON,
        storage_path=str(tmp_path),
        interval_minutes=15,
        forecast_type=MLForecastType.SOLAR,
        backend=MLBackendType.NIXTLA_HYBRID,
        preserve_gaps=True,
        hybrid_structure=HybridStructure.IRRADIANCE_DIRECT,
    )

    def fetch_historical(start: object, end: object, **_: object) -> pd.DataFrame:
        days = weather["ds"].dt.date
        return weather.loc[(days >= start) & (days <= end)].reset_index(drop=True)

    def fetch_forecast(days: int, past_days: int = 0, **_: object) -> pd.DataFrame:
        today = pd.Timestamp.now(tz="UTC").floor("D")
        start = today - pd.Timedelta(days=past_days)
        end = today + pd.Timedelta(days=days)
        rows = (weather["ds"] >= start) & (weather["ds"] < end)
        frame = weather.loc[rows].reset_index(drop=True)
        # Forecast weather never carries the past snowfall: the load on the
        # panels has to come from the training history.
        frame.loc[frame["ds"] <= snow_free_until, "snowfall"] = 0.0
        return frame

    client = forecast.weather_client
    monkeypatch.setattr(client, "fetch_historical", fetch_historical)
    monkeypatch.setattr(client, "fetch_forecast", fetch_forecast)
    monkeypatch.setattr(client, "resample_weather", lambda frame, _: frame)
    return forecast


def _history_end() -> pd.Timestamp:
    """Half an hour after yesterday's sunset, the hardest point to retrain at.

    A covered day then ends in a few measured night rows after hours of
    held-out daylight, and selection has the least measured tail to use.
    """
    yesterday = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
    ds = pd.date_range(yesterday, periods=96, freq="15min")
    ghi = get_clear_sky_ghi(_LAT, _LON, pd.DatetimeIndex(ds))
    return ds[ghi.to_numpy() > 0][-1] + pd.Timedelta(minutes=30)


def _weather(history_end: pd.Timestamp) -> pd.DataFrame:
    ds = pd.date_range(
        history_end - pd.Timedelta(days=6),
        history_end + pd.Timedelta(days=4),
        freq="15min",
    )
    return pd.DataFrame(
        {
            "ds": ds,
            "temperature_2m": 4.0,
            "cloud_cover": 0.0,
            "shortwave_radiation": get_clear_sky_ghi(
                _LAT, _LON, pd.DatetimeIndex(ds)
            ).to_numpy(),
            "snow_depth": 0.0,
            "snowfall": 0.0,
        }
    )


def _snow_from(weather: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """Add a 12 cm cold snowfall from ``start``; return the rows from then on."""
    after = weather["ds"] >= start
    weather.loc[after & (weather["ds"] < start + pd.Timedelta(hours=4)), "snowfall"] = (
        3.0  # cm/h
    )
    weather.loc[after, "temperature_2m"] = -3.0
    return after


@pytest.mark.parametrize("snow", [True, False])
def test_snow_in_training_history_zeroes_the_next_forecast(
    tmp_path, monkeypatch, snow
) -> None:
    history_end = _history_end()
    weather = _weather(history_end)
    y = weather["shortwave_radiation"] * _KWH_PER_W_M2
    if snow:
        after_snow = _snow_from(weather, history_end - pd.Timedelta(hours=40))
        y = y.where(~after_snow, 0.0)
    history = pd.DataFrame({"ds": weather["ds"], "y": y}).loc[
        lambda f: f["ds"] <= history_end
    ]

    trained = _site(tmp_path, monkeypatch, weather, history_end)
    trained.train(history, force=True)
    # A fresh instance serves from the stored manifest, as after a restart.
    served = _site(tmp_path, monkeypatch, weather, history_end)
    forecast = served.predict(days=1)

    # The cutoff stays at the last row although the covered day before it is
    # held out of the fit.
    assert served.training_end == history_end.to_pydatetime()
    assert served.snow_cover is not None
    if snow:
        assert trained.snow_excluded_rows > 0
        assert served.snow_cover["status"] == "covered"
        assert forecast.filter(like="yhat").to_numpy().max() == 0.0
    else:
        assert trained.snow_excluded_rows == 0
        assert served.snow_cover["status"] == "clear"
        assert forecast["yhat"].max() > 0.5


def test_snow_in_the_forecast_weather_zeroes_output_after_it_falls(
    tmp_path, monkeypatch
) -> None:
    history_end = _history_end()
    weather = _weather(history_end)
    snow_start = history_end.floor("D") + pd.Timedelta(days=2)
    _snow_from(weather, snow_start)
    history = pd.DataFrame(
        {"ds": weather["ds"], "y": weather["shortwave_radiation"] * _KWH_PER_W_M2}
    ).loc[lambda f: f["ds"] <= history_end]

    trained = _site(tmp_path, monkeypatch, weather, history_end)
    trained.train(history, force=True)
    origin = history_end + pd.Timedelta(minutes=15)
    forecast = trained.predict(days=2, origin=origin.to_pydatetime())

    assert trained.snow_excluded_rows == 0
    covered = pd.to_datetime(forecast["ds"], utc=True) >= snow_start
    assert covered.any() and (~covered).any()
    assert forecast.loc[~covered, "yhat"].max() > 0.5
    assert forecast.loc[covered].filter(like="yhat").to_numpy().max() == 0.0


def test_measured_output_clears_a_window_that_starts_under_snow(
    tmp_path, monkeypatch
) -> None:
    history_end = _history_end()
    history_start = history_end - pd.Timedelta(days=3)
    weather = _weather(history_end)
    # 40 cm at the start of the window: the weather alone sheds none of it
    # in three days, so no row is clear by the weather.
    _snow_from(weather, history_start)
    weather.loc[
        (weather["ds"] >= history_start)
        & (weather["ds"] < history_start + pd.Timedelta(hours=4)),
        "snowfall",
    ] = 10.0
    cleared = history_start + pd.Timedelta(days=1)
    y = (weather["shortwave_radiation"] * _KWH_PER_W_M2).where(
        weather["ds"] >= cleared, 0.0
    )
    history = pd.DataFrame({"ds": weather["ds"], "y": y}).loc[
        lambda f: (f["ds"] >= history_start) & (f["ds"] <= history_end)
    ]

    trained = _site(tmp_path, monkeypatch, weather, history_end)
    trained.train(history, force=True)
    forecast = trained.predict(
        days=1, origin=(history_end + pd.Timedelta(minutes=15)).to_pydatetime()
    )

    # Only the covered first day is held out, plus up to an hour of output
    # it takes to see the panels clear; the produced days then prove them
    # clear and the forecast follows them.
    covered_daylight = history["ds"].lt(cleared) & weather.loc[history.index][
        "shortwave_radiation"
    ].gt(0)
    assert 0 < trained.snow_excluded_rows <= int(covered_daylight.sum()) + 4
    assert trained.snow_cover is not None
    assert trained.snow_cover["status"] == "clear"
    assert forecast["yhat"].max() > 0.5
