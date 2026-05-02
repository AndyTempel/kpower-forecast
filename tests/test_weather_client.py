import pandas as pd

from kpower_forecast.weather_client import WeatherClient


def test_process_response_converts_naive_local_times_to_utc():
    client = WeatherClient(lat=46.0, lon=14.0)
    data = {
        "timezone": "Europe/Ljubljana",
        "utc_offset_seconds": 7200,
        "hourly": {
            "time": ["2024-06-01T02:00", "2024-06-01T03:00"],
            "temperature_2m": [20.0, 21.0],
            "cloud_cover": [0.0, 0.0],
            "shortwave_radiation": [0.0, 10.0],
            "snow_depth": [None, None],
            "snowfall": [None, None],
        },
    }

    df = client._process_response(data)

    assert df["ds"].tolist() == [
        pd.Timestamp("2024-06-01T00:00:00Z"),
        pd.Timestamp("2024-06-01T01:00:00Z"),
    ]


def test_process_response_keeps_gmt_times_as_utc():
    client = WeatherClient(lat=46.0, lon=14.0)
    data = {
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly": {
            "time": ["2024-06-01T00:00", "2024-06-01T01:00"],
            "temperature_2m": [20.0, 21.0],
            "cloud_cover": [0.0, 0.0],
            "shortwave_radiation": [0.0, 10.0],
            "snow_depth": [None, None],
            "snowfall": [None, None],
        },
    }

    df = client._process_response(data)

    assert df["ds"].tolist() == [
        pd.Timestamp("2024-06-01T00:00:00Z"),
        pd.Timestamp("2024-06-01T01:00:00Z"),
    ]
