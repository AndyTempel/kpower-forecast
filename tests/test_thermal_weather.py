"""UTC weather alignment for real sparse thermal observations."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

from kpower_forecast.thermal import (
    KPowerThermalForecast,
    ThermalObservedTransition,
)
from kpower_forecast.thermal.weather import (
    exact_future_outdoor_grid,
    mean_outdoor_temperature,
)
from kpower_forecast.weather_client import WeatherClient, WeatherConfig


def _weather() -> pd.DataFrame:
    """Return ordered quarter-hour weather with a linear Celsius ramp."""
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01T00:00:00Z", periods=9, freq="15min"),
            "temperature_2m": list(range(9)),
        }
    )


def test_integrates_irregular_observation_span_without_target_grid() -> None:
    """Outdoor mean uses exact elapsed time between two indoor observations."""
    result = mean_outdoor_temperature(
        _weather(),
        start_at=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        end_at=datetime(2026, 1, 1, 1, 20, tzinfo=timezone.utc),
    )
    assert result == pytest.approx(3.0)


def test_weather_gap_and_missing_future_slot_fail() -> None:
    """Material weather gaps cannot silently acquire numerical coverage."""
    frame = _weather().drop(index=[1, 2, 3, 4]).reset_index(drop=True)
    with pytest.raises(ValueError, match="gap"):
        mean_outdoor_temperature(
            frame,
            start_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_at=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
            max_sample_gap=timedelta(minutes=30),
        )
    with pytest.raises(ValueError, match="exact UTC"):
        exact_future_outdoor_grid(
            frame,
            origin=datetime(2026, 1, 1, tzinfo=timezone.utc),
            periods=5,
            interval_minutes=15,
        )


def test_exact_weather_endpoints_ignore_gaps_outside_transition() -> None:
    """Only weather samples required inside an interval gate its coverage."""
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                    "2026-01-01T07:00:00Z",
                ],
                utc=True,
            ),
            "temperature_2m": [0.0, 2.0, 3.0, 7.0],
        }
    )
    assert mean_outdoor_temperature(
        frame,
        start_at=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
        end_at=datetime(2026, 1, 1, 3, tzinfo=timezone.utc),
    ) == pytest.approx(2.5)


def test_exact_future_weather_grid_across_dst() -> None:
    """Canonical UTC forecast timestamps remain continuous across local DST."""
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-03-29T00:00:00Z", periods=8, freq="15min"),
            "temperature_2m": list(range(8)),
        }
    )
    values = exact_future_outdoor_grid(
        frame,
        origin=datetime(2026, 3, 29, tzinfo=timezone.utc),
        periods=8,
        interval_minutes=15,
    )
    assert values == list(range(8))


def test_package_weather_adapter_keeps_one_real_transition(tmp_path: object) -> None:
    """The high-level adapter fetches strict weather without target filling."""
    weather = Mock(spec=WeatherClient)
    weather.lat = 46.0
    weather.lon = 14.0
    weather.config = WeatherConfig()
    weather.fetch_historical.return_value = _weather()
    model = KPowerThermalForecast(
        model_id="zone_kitchen",
        authority_fingerprint="epoch-a",
        storage_path=tmp_path,
        weather_client=weather,
    )
    transition = ThermalObservedTransition(
        start_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        indoor_start_c=19,
        indoor_end_c=19.1,
        hvac_electric_mean_w=1000,
        hvac_coverage_ratio=1.0,
    )
    diagnostics = model.train_with_weather([transition])
    assert diagnostics.transition_count == 1
    assert diagnostics.unreliable_reason == "insufficient_history"
    assert weather.fetch_historical.call_args.kwargs["strict"] is True
