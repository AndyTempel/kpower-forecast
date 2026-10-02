import datetime
import logging
from pathlib import Path

import pandas as pd
import pytest
import requests

from kpower_forecast.weather_client import WeatherClient, WeatherConfig


def test_recent_training_weather_caps_future_padding_and_shares_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Thermal and electrical clients reuse valid archive and recent weather."""
    clock = datetime.datetime(2026, 9, 28, 15, 15, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(
        "kpower_forecast.weather_client._weather_today", lambda: clock.date()
    )
    calls: list[tuple[str, dict[str, object]]] = []

    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self.payload

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        calls.append((url, params.copy()))
        if "archive" in url:
            assert params["start_date"] == "2026-09-23"
            assert params["end_date"] == "2026-09-26"
            return Response(_weather_payload(["2026-09-26T23:45"], [10.0]))
        assert params["past_days"] == 1
        assert params["forecast_days"] == 1
        return Response(
            _weather_payload(
                ["2026-09-27T00:00", "2026-09-28T15:00", "2026-09-28T15:15"],
                [11.0, None, 13.0],
            )
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)
    config = WeatherConfig(cache_dir=tmp_path, long_horizon_model=None)
    thermal = WeatherClient(46, 14, config)
    electrical = WeatherClient(46, 14, config)
    first = thermal.fetch_historical(
        datetime.date(2026, 9, 23), datetime.date(2026, 9, 29), strict=True
    )
    second = electrical.fetch_historical(
        datetime.date(2026, 9, 23), datetime.date(2026, 9, 28), strict=True
    )
    assert len(calls) == 2
    pd.testing.assert_frame_equal(first, second)
    assert first["temperature_2m"].isna().tolist() == [False, False, True, False]
    assert first["ds"].max() == pd.Timestamp("2026-09-28T15:15Z")


def test_future_only_history_does_not_make_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future-only training ranges fail without an invalid archive request."""
    monkeypatch.setattr(
        "kpower_forecast.weather_client._weather_today",
        lambda: datetime.date(2026, 9, 28),
    )
    client = WeatherClient(46, 14)
    with pytest.raises(ValueError, match="entirely in the future"):
        client.fetch_historical(datetime.date(2026, 9, 29), datetime.date(2026, 9, 30))


def test_recent_only_history_skips_archive_and_excludes_later_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yesterday-only request uses recent weather but returns no today's rows."""
    monkeypatch.setattr(
        "kpower_forecast.weather_client._weather_today",
        lambda: datetime.date(2026, 9, 28),
    )
    calls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(
                ["2026-09-27T23:45", "2026-09-28T00:00"], [11.0, 12.0]
            )

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        calls.append(url)
        assert params["past_days"] == 1
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)
    client = WeatherClient(
        46, 14, WeatherConfig(cache_enabled=False, long_horizon_model=None)
    )
    result = client.fetch_historical(
        datetime.date(2026, 9, 27), datetime.date(2026, 9, 27), strict=True
    )
    assert calls == [client.config.base_url]
    assert result["ds"].tolist() == [pd.Timestamp("2026-09-27T23:45Z")]


def test_forecast_cache_does_not_replay_previous_utc_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Relative day queries refresh across midnight even inside their TTL."""
    clock = datetime.datetime(2026, 9, 28, 23, 45, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr(
        "kpower_forecast.weather_client._weather_today", lambda: clock.date()
    )
    calls = 0

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(
                [clock.date().isoformat() + "T00:00"], [float(calls)]
            )

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)
    client = WeatherClient(
        46, 14, WeatherConfig(cache_dir=tmp_path, long_horizon_model=None)
    )
    client.fetch_forecast(days=1)
    clock += datetime.timedelta(minutes=15)
    result = client.fetch_forecast(days=1)
    assert calls == 2
    assert result["ds"].min() == pd.Timestamp("2026-09-29T00:00Z")


def test_forecast_response_crossing_midnight_keeps_original_cache_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An in-flight yesterday response cannot poison today's cache entry."""
    day = datetime.date(2026, 9, 28)
    monkeypatch.setattr("kpower_forecast.weather_client._weather_today", lambda: day)
    calls = 0

    class Response:
        def __init__(self, date: datetime.date) -> None:
            self.date = date

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload([self.date.isoformat() + "T00:00"], [10.0])

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        nonlocal calls, day
        calls += 1
        response = Response(day)
        if calls == 1:
            day += datetime.timedelta(days=1)
        return response

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)
    client = WeatherClient(
        46, 14, WeatherConfig(cache_dir=tmp_path, long_horizon_model=None)
    )
    first = client.fetch_forecast(days=1)
    second = client.fetch_forecast(days=1)
    assert calls == 2
    assert first["ds"].min() == pd.Timestamp("2026-09-28T00:00Z")
    assert second["ds"].min() == pd.Timestamp("2026-09-29T00:00Z")


def test_weather_config_defaults_recent_forecast_history_to_one_day() -> None:
    assert WeatherConfig().recent_forecast_past_days == 1


def _weather_payload(
    times: list[str],
    temperature: list[float | None],
    payload_key: str = "minutely_15",
) -> dict[str, object]:
    """Build a minimal Open-Meteo weather response payload."""
    return {
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        payload_key: {
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

    client.fetch_forecast(days=5, past_days=2)

    assert "models" not in observed_params
    assert observed_params["forecast_days"] == 5
    assert observed_params["past_days"] == 2
    weather_variables = observed_params["minutely_15"]
    assert isinstance(weather_variables, list)
    assert "direct_radiation" in weather_variables


def test_strict_forecast_keeps_missing_outdoor_temperature(monkeypatch) -> None:
    """Thermal callers can reject weather gaps before any normal fill occurs."""
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False, long_horizon_model=None),
    )

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T00:15"], [10.0, None]
            )

    monkeypatch.setattr(
        "kpower_forecast.weather_client.requests.get",
        lambda *args, **kwargs: Response(),
    )
    strict = client.fetch_forecast(days=1, strict=True)
    assert pd.isna(strict.loc[1, "temperature_2m"])
    normal = client.fetch_forecast(days=1)
    assert normal.loc[1, "temperature_2m"] == 10.0


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
                    "2026-05-01T00:00", periods=480, freq="15min"
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
                        "2026-05-01T00:15",
                        "2026-05-01T00:30",
                        "2026-05-01T00:45",
                    ],
                    [100.0, 101.0, 102.0, 103.0],
                )
            )
        return Response(
            _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T00:15"],
                [10.0, None],
            )
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    frame = client.fetch_forecast(days=1)

    assert observed_models == [None, "ecmwf_ifs"]
    assert frame["ds"].tolist() == [
        pd.Timestamp("2026-05-01T00:00:00Z"),
        pd.Timestamp("2026-05-01T00:15:00Z"),
        pd.Timestamp("2026-05-01T00:30:00Z"),
        pd.Timestamp("2026-05-01T00:45:00Z"),
    ]
    assert frame["temperature_2m"].tolist() == [10.0, 101.0, 102.0, 103.0]
    assert "Returning partial weather data" in caplog.text


def test_fetch_forecast_fills_empty_best_match_from_long_model(
    monkeypatch, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="kpower_forecast.weather_client")
    client = WeatherClient(
        lat=40.7128,
        lon=-74.0060,
        config=WeatherConfig(cache_enabled=False),
    )
    observed_models: list[object] = []
    times = [
        timestamp.strftime("%Y-%m-%dT%H:%M")
        for timestamp in pd.date_range("2026-05-01T00:00", periods=96, freq="15min")
    ]

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
            return Response(_weather_payload(times, [20.0] * len(times)))
        payload = _weather_payload(times, [None] * len(times))
        weather_payload = payload["minutely_15"]
        assert isinstance(weather_payload, dict)
        weather_payload["cloud_cover"] = [None] * len(times)
        weather_payload["shortwave_radiation"] = [None] * len(times)
        return Response(payload)

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    frame = client.fetch_forecast(days=1)

    assert observed_models == [None, "ecmwf_ifs"]
    assert len(frame) == 96
    assert frame["temperature_2m"].isna().sum() == 0
    assert frame["temperature_2m"].tolist() == [20.0] * 96
    assert "complete required weather data until none" in caplog.text


def test_fetch_forecast_fills_null_padded_tail_from_long_model(
    monkeypatch, caplog
) -> None:
    # Live shape: the 15-min primary ends about 2.5 days out and pads the rest
    # of a longer horizon with nulls. A successful fill is routine, not a warning.
    caplog.set_level(logging.INFO, logger="kpower_forecast.weather_client")
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False),
    )
    observed_models: list[object] = []
    times = [
        timestamp.strftime("%Y-%m-%dT%H:%M")
        for timestamp in pd.date_range("2026-05-01T00:00", periods=96, freq="15min")
    ]

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
        model = params.get("models")
        observed_models.append(model)
        if model == "ecmwf_ifs":
            return Response(_weather_payload(times, [20.0] * len(times)))

        payload = _weather_payload(times, [10.0] * 4 + [None] * 92)
        weather = payload["minutely_15"]
        assert isinstance(weather, dict)
        weather["cloud_cover"] = [20.0] * 4 + [None] * 92
        weather["shortwave_radiation"] = [50.0] * 4 + [None] * 92
        return Response(payload)

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    frame = client.fetch_forecast(days=1)

    assert observed_models == [None, "ecmwf_ifs"]
    assert len(frame) == 96
    assert frame.loc[3, "shortwave_radiation"] == 50.0
    assert frame.loc[4, "shortwave_radiation"] == 50.0
    assert frame.loc[4, "temperature_2m"] == 20.0
    complete_columns = frame[["temperature_2m", "cloud_cover", "shortwave_radiation"]]
    assert complete_columns.notna().all().all()
    assert "until 2026-05-01T00:45:00+00:00" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_fetch_forecast_warns_when_long_model_leaves_required_gaps(
    monkeypatch, caplog
) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False),
    )
    times = [
        timestamp.strftime("%Y-%m-%dT%H:%M")
        for timestamp in pd.date_range("2026-05-01T00:00", periods=96, freq="15min")
    ]

    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        # Both models stop after one hour of data.
        return Response(_weather_payload(times, [10.0] * 4 + [None] * 92))

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)

    client.fetch_forecast(days=1)

    assert any(
        record.levelno == logging.WARNING
        and "did not complete the required weather data" in record.getMessage()
        for record in caplog.records
    )


def test_process_response_prefers_minutely_15_payload() -> None:
    client = WeatherClient(lat=46.0, lon=14.0)
    data = {
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "hourly": {
            "time": ["2024-06-01T00:00"],
            "temperature_2m": [99.0],
        },
        "minutely_15": {
            "time": ["2024-06-01T00:00", "2024-06-01T00:15"],
            "temperature_2m": [20.0, 21.0],
            "cloud_cover": [100.0, 90.0],
            "shortwave_radiation": [0.0, 10.0],
            "snow_depth": [None, None],
            "snowfall": [None, None],
        },
    }

    df = client._process_response(data)

    assert df["ds"].tolist() == [
        pd.Timestamp("2024-06-01T00:00:00Z"),
        pd.Timestamp("2024-06-01T00:15:00Z"),
    ]
    assert df["temperature_2m"].tolist() == [20.0, 21.0]


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
    observed_weather_params: list[list[str]] = []
    observed_request_fields: list[str] = []
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
        weather = params.get("minutely_15")
        assert isinstance(weather, list)
        observed_request_fields.append("minutely_15")
        observed_weather_params.append(list(weather))
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

    assert observed_request_fields == ["minutely_15", "minutely_15"]
    assert len(observed_weather_params) == 2
    assert "snowfall_convective_water_equivalent" in observed_weather_params[0]
    assert "snowfall_convective_water_equivalent" not in observed_weather_params[1]
    assert not frame.empty


def test_fetch_historical_falls_back_to_hourly_when_minutely_15_is_empty(
    monkeypatch, caplog
) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False),
    )
    observed_request_fields: list[str] = []

    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: float) -> Response:
        assert timeout == 10
        assert url == "https://archive-api.open-meteo.com/v1/archive"
        if "minutely_15" in params:
            observed_request_fields.append("minutely_15")
            return Response(
                {
                    "latitude": 46.08084,
                    "longitude": 14.45151,
                    "utc_offset_seconds": 0,
                    "timezone": "GMT",
                    "timezone_abbreviation": "GMT",
                    "elevation": 302.0,
                }
            )

        observed_request_fields.append("hourly")
        hourly = params.get("hourly")
        assert isinstance(hourly, list)
        return Response(
            _weather_payload(
                ["2026-06-30T00:00", "2026-06-30T01:00"],
                [21.2, 20.5],
                payload_key="hourly",
            )
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)
    caplog.set_level(logging.INFO)

    frame = client.fetch_historical(
        start_date=pd.Timestamp("2026-06-30").date(),
        end_date=pd.Timestamp("2026-06-30").date(),
    )

    assert observed_request_fields == ["minutely_15", "hourly"]
    assert frame["temperature_2m"].tolist() == [21.2, 20.5]
    assert "Retrying with hourly data" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


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
    observed_weather_params: list[list[str]] = []
    observed_request_fields: list[str] = []
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
        weather = params.get("minutely_15")
        assert isinstance(weather, list)
        observed_request_fields.append("minutely_15")
        observed_weather_params.append(list(weather))
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

    assert observed_request_fields == ["minutely_15", "minutely_15"]
    assert len(observed_weather_params) == 2
    assert "snowfall_convective_water_equivalent" in observed_weather_params[0]
    assert "snowfall_convective_water_equivalent" not in observed_weather_params[1]
    assert not frame.empty


def test_fetch_forecast_falls_back_to_hourly_when_minutely_15_is_rejected(
    monkeypatch, caplog
) -> None:
    client = WeatherClient(
        lat=46.0,
        lon=14.0,
        config=WeatherConfig(cache_enabled=False, long_horizon_model=None),
    )
    observed_request_fields: list[str] = []

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
        assert timeout == 10
        assert url == "https://api.open-meteo.com/v1/forecast"
        if "minutely_15" in params:
            observed_request_fields.append("minutely_15")
            return Response(400, {"reason": "unsupported parameter", "error": True})

        observed_request_fields.append("hourly")
        hourly = params.get("hourly")
        assert isinstance(hourly, list)
        return Response(
            200,
            _weather_payload(
                ["2026-05-01T00:00", "2026-05-01T01:00"],
                [10.0, 11.0],
                payload_key="hourly",
            ),
        )

    monkeypatch.setattr("kpower_forecast.weather_client.requests.get", fake_get)
    caplog.set_level(logging.INFO)

    frame = client.fetch_forecast(days=1)

    assert observed_request_fields == ["minutely_15", "hourly"]
    assert frame["temperature_2m"].tolist() == [10.0, 11.0]
    assert "Retrying with hourly data" in caplog.text
    assert not any(
        record.levelno >= logging.WARNING
        and "Retrying with hourly data" in record.getMessage()
        for record in caplog.records
    )


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
