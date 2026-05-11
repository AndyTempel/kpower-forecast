import pandas as pd

from kpower_forecast.weather_client import WeatherClient, WeatherConfig


def test_fetch_forecast_requests_ecmwf_ifs_model(monkeypatch) -> None:
    client = WeatherClient(lat=46.0, lon=14.0)
    observed_params: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "timezone": "UTC",
                "utc_offset_seconds": 0,
                "hourly": {
                    "time": ["2026-05-01T00:00", "2026-05-01T01:00"],
                    "temperature_2m": [10.0, 11.0],
                    "cloud_cover": [20.0, 25.0],
                    "shortwave_radiation": [0.0, 50.0],
                    "snow_depth": [None, None],
                    "snowfall": [None, None],
                },
            }

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        observed_params.update(params)
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert timeout == 10
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    client.fetch_forecast(days=5)

    assert observed_params["models"] == "ecmwf_ifs"
    assert observed_params["forecast_days"] == 5
    hourly_variables = observed_params["hourly"]
    assert isinstance(hourly_variables, list)
    assert "direct_radiation" in hourly_variables


def test_process_response_includes_available_optional_fields() -> None:
    client = WeatherClient(lat=46.0, lon=14.0)
    data = {
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "hourly": {
            "time": ["2024-06-01T00:00", "2024-06-01T01:00"],
            "temperature_2m": [20.0, 21.0],
            "cloud_cover": [100.0, 90.0],
            "shortwave_radiation": [0.0, 10.0],
            "snow_depth": [None, None],
            "snowfall": [None, None],
            "direct_radiation": [0.0, 2.0],
            "diffuse_radiation": [0.0, 8.0],
            "rain": [0.0, 1.0],
        },
    }

    df = client._process_response(data)

    assert df["direct_radiation"].tolist() == [0.0, 2.0]
    assert df["diffuse_radiation"].tolist() == [0.0, 8.0]
    assert df["rain"].tolist() == [0.0, 1.0]


def test_process_response_omits_unavailable_optional_fields() -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(optional_hourly_variables=["direct_radiation"]),
    )
    data = {
        "timezone": "UTC",
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

    assert "direct_radiation" not in df.columns


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


def test_resample_weather_clips_physical_negative_values() -> None:
    client = WeatherClient(lat=46.0, lon=14.0)
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-01", periods=4, freq="h", tz="UTC"),
            "temperature_2m": [10.0, 12.0, 11.0, 13.0],
            "shortwave_radiation": [0.0, -10.0, 50.0, 0.0],
            "direct_radiation": [0.0, -5.0, 40.0, 0.0],
            "diffuse_radiation": [0.0, -3.0, 10.0, 0.0],
            "snow_depth": [0.0, -0.1, 0.0, 0.0],
            "rain": [0.0, -1.0, 0.0, 0.0],
        }
    )

    resampled = client.resample_weather(df, interval_minutes=15)

    for column in [
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "snow_depth",
        "rain",
    ]:
        assert resampled[column].min() >= 0.0
