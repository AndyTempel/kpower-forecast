import pandas as pd
import requests

from kpower_forecast.weather_client import WeatherClient, WeatherConfig


def _weather_payload(
    times: list[str], temperature: list[float | None]
) -> dict[str, object]:
    """Build a minimal Open-Meteo hourly response payload."""
    return {
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "hourly": {
            "time": times,
            "temperature_2m": temperature,
            "cloud_cover": [20.0] * len(times),
            "shortwave_radiation": [50.0] * len(times),
            "snow_depth": [None] * len(times),
            "snowfall": [None] * len(times),
        },
    }


def test_fetch_forecast_omits_model_by_default(monkeypatch) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False, long_horizon_model=None),
    )
    observed_params: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T01:00"], [10.0, 11.0]
            )

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        observed_params.update(params)
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert timeout == 10
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    client.fetch_forecast(days=5)

    assert "models" not in observed_params
    assert observed_params["forecast_days"] == 5
    hourly_variables = observed_params["hourly"]
    assert isinstance(hourly_variables, list)
    assert "direct_radiation" in hourly_variables


def test_fetch_forecast_uses_explicit_primary_model(monkeypatch) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(
            forecast_model="dwd_icon_d2",
            cache_enabled=False,
        ),
    )
    observed_params: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            times = [
                timestamp.strftime("%Y-%m-%dT%H:%M")
                for timestamp in pd.date_range(
                    "2026-05-01T00:00", periods=120, freq="h"
                )
            ]
            return _weather_payload(
                times,
                [10.0] * len(times),
            )

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        observed_params.update(params)
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert timeout == 10
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    client.fetch_forecast(days=5)

    assert observed_params["models"] == "dwd_icon_d2"
    assert client.effective_forecast_model_id() == "dwd_icon_d2+ecmwf_ifs"


def test_fetch_forecast_fills_short_horizon_from_long_model(
    monkeypatch, caplog
) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False),
    )
    observed_models: list[object] = []

    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert timeout == 10
        observed_models.append(params.get("models"))
        if params.get("models") == "ecmwf_ifs":
            return Response(
                _weather_payload(
                    [
                        "2026-05-01T00:00",
                        "2026-05-01T01:00",
                        "2026-05-01T02:00",
                        "2026-05-01T03:00",
                    ],
                    [100.0, 101.0, 102.0, 103.0],
                )
            )
        return Response(
            _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T01:00"],
                [10.0, None],
            )
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    frame = client.fetch_forecast(days=1)

    assert observed_models == [None, "ecmwf_ifs"]
    assert frame["ds"].tolist() == [
        pd.Timestamp("2026-05-01T00:00:00Z"),
        pd.Timestamp("2026-05-01T01:00:00Z"),
        pd.Timestamp("2026-05-01T02:00:00Z"),
        pd.Timestamp("2026-05-01T03:00:00Z"),
    ]
    assert frame["temperature_2m"].tolist() == [10.0, 101.0, 102.0, 103.0]
    assert "Returning partial weather data" in caplog.text


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


def test_fetch_historical_retries_without_invalid_archive_variable(monkeypatch) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(
            optional_hourly_variables=["snowfall_convective_water_equivalent"],
            cache_enabled=False,
        ),
    )
    observed_hourly_params: list[list[str]] = []
    call_count = 0

    class Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError("bad request", response=self)

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal call_count
        assert timeout == 10
        assert url == "https://archive-api.open-meteo.com/v1/archive"
        hourly = params.get("hourly")
        assert isinstance(hourly, list)
        observed_hourly_params.append(list(hourly))
        call_count += 1

        if call_count == 1:
            return Response(
                400,
                {
                    "reason": (
                        "Data corrupted at path ''. Cannot initialize "
                        "SurfacePressureAndHeightVariable<...> from invalid String "
                        "value snowfall_convective_water_equivalent."
                    ),
                    "error": True,
                },
            )

        return Response(
            200,
            {
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
            },
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    frame = client.fetch_historical(
        start_date=pd.Timestamp("2026-05-01").date(),
        end_date=pd.Timestamp("2026-05-01").date(),
    )

    assert len(observed_hourly_params) == 2
    assert "snowfall_convective_water_equivalent" in observed_hourly_params[0]
    assert "snowfall_convective_water_equivalent" not in observed_hourly_params[1]
    assert not frame.empty


def test_fetch_forecast_retries_without_invalid_variable(monkeypatch) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(
            optional_hourly_variables=["snowfall_convective_water_equivalent"],
            cache_enabled=False,
            long_horizon_model=None,
        ),
    )
    observed_hourly_params: list[list[str]] = []
    call_count = 0

    class Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError("bad request", response=self)

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal call_count
        assert timeout == 10
        assert url == "https://api.open-meteo.com/v1/forecast"
        hourly = params.get("hourly")
        assert isinstance(hourly, list)
        observed_hourly_params.append(list(hourly))
        call_count += 1

        if call_count == 1:
            return Response(
                400,
                {
                    "reason": (
                        "invalid String value snowfall_convective_water_equivalent"
                    ),
                    "error": True,
                },
            )

        return Response(
            200,
            {
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
            },
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    frame = client.fetch_forecast(days=2)

    assert len(observed_hourly_params) == 2
    assert "snowfall_convective_water_equivalent" in observed_hourly_params[0]
    assert "snowfall_convective_water_equivalent" not in observed_hourly_params[1]
    assert not frame.empty


def test_fetch_forecast_uses_cache_for_repeated_payload(monkeypatch, tmp_path) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_dir=tmp_path, long_horizon_model=None),
    )
    call_count = 0

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T01:00"], [10.0, 11.0]
            )

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal call_count
        assert timeout == 10
        call_count += 1
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    first = client.fetch_forecast(days=1)
    second = client.fetch_forecast(days=1)

    assert call_count == 1
    assert first.equals(second)


def test_fetch_historical_uses_cache_for_repeated_payload(
    monkeypatch, tmp_path
) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_dir=tmp_path),
    )
    call_count = 0

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T01:00"], [10.0, 11.0]
            )

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal call_count
        assert url == "https://archive-api.open-meteo.com/v1/archive"
        assert timeout == 10
        call_count += 1
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    first = client.fetch_historical(
        start_date=pd.Timestamp("2026-05-01").date(),
        end_date=pd.Timestamp("2026-05-01").date(),
    )
    second = client.fetch_historical(
        start_date=pd.Timestamp("2026-05-01").date(),
        end_date=pd.Timestamp("2026-05-01").date(),
    )

    assert call_count == 1
    assert first.equals(second)


def test_expired_forecast_cache_is_refreshed(monkeypatch, tmp_path) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(
            cache_dir=tmp_path,
            forecast_cache_ttl_hours=0.000000000001,
            long_horizon_model=None,
        ),
    )
    call_count = 0

    class Response:
        def __init__(self, value: float) -> None:
            self.value = value

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(["2026-05-01T00:00"], [self.value])

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal call_count
        assert timeout == 10
        call_count += 1
        return Response(float(call_count))

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    first = client.fetch_forecast(days=1)
    second = client.fetch_forecast(days=1)

    assert call_count == 2
    assert first["temperature_2m"].tolist() == [1.0]
    assert second["temperature_2m"].tolist() == [2.0]
