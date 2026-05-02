import pandas as pd
import pytest

from kpower_forecast.utils import convert_units, normalize_to_instant_kwh


def test_convert_units():
    df = pd.DataFrame({"y": [1000, 2000]})

    # W to kWh
    res = convert_units(df, from_unit="W", to_unit="kWh")
    assert res["y"].tolist() == [1.0, 2.0]

    # kW to kWh (same base, should be 1:1)
    res = convert_units(df, from_unit="kW", to_unit="kWh")
    assert res["y"].tolist() == [1000.0, 2000.0]

    # Wh to kWh
    res = convert_units(df, from_unit="Wh", to_unit="kWh")
    assert res["y"].tolist() == [1.0, 2.0]


def test_normalize_cumulative_energy():
    # Cumulative meter readings (kWh)
    # Reading at 10:00: 100
    # Reading at 10:15: 105.5 -> diff 5.5
    # Reading at 10:30: 110.0 -> diff 4.5
    data = {
        "ds": ["2024-01-01 10:00", "2024-01-01 10:15", "2024-01-01 10:30"],
        "y": [100.0, 105.5, 110.0],
    }
    df = pd.DataFrame(data)

    res = normalize_to_instant_kwh(
        df, category="cumulative_energy", unit="kWh", target_interval_min=15
    )

    # With Index Union interpolation:
    # 10:00 is first point. diff() at 10:15 should capture growth from 10:00.
    assert any(val == pytest.approx(5.5) for val in res["y"].values)
    assert any(val == pytest.approx(4.5) for val in res["y"].values)


def test_normalize_power_inconsistent_intervals():
    # Power readings (kW) at inconsistent times
    # 10:00: 10kW
    # 10:06: 20kW
    #   Integration: (Reading at 10:06) * (6 min) = 20 * 0.1 = 2.0kWh
    # 10:15: 10kW
    #   Integration: (Reading at 10:15) * (9 min) = 10 * 0.15 = 1.5kWh
    # Total kWh = 3.5
    data = {
        "ds": ["2024-01-01 10:00", "2024-01-01 10:06", "2024-01-01 10:15"],
        "y": [10.0, 20.0, 10.0],
    }
    df = pd.DataFrame(data)

    res = normalize_to_instant_kwh(
        df, category="power", unit="kW", target_interval_min=15
    )

    # Total should stay aligned with the current timestamp label.
    assert res["y"].sum() == pytest.approx(3.5)


def test_normalize_power_uses_current_sample_for_interval_label():
    data = {
        "ds": [
            "2024-01-01 06:45+00:00",
            "2024-01-01 07:00+00:00",
            "2024-01-01 07:15+00:00",
        ],
        "y": [0.0, 1.0, 2.0],
    }
    df = pd.DataFrame(data)

    res = normalize_to_instant_kwh(
        df, category="power", unit="kW", target_interval_min=15
    )

    actual = res.loc[
        res["ds"] == pd.Timestamp("2024-01-01 07:00", tz="UTC"), "y"
    ].item()
    assert actual == pytest.approx(0.25)


def test_normalize_instant_energy_resampling():
    # Instant readings (Wh) every 5 minutes
    # 10:00: 100Wh
    # 10:05: 100Wh
    # 10:10: 100Wh
    # 10:15: 100Wh
    # Total = 400Wh = 0.4kWh
    data = {
        "ds": [
            "2024-01-01 10:00",
            "2024-01-01 10:05",
            "2024-01-01 10:10",
            "2024-01-01 10:15",
        ],
        "y": [100, 100, 100, 100],
    }
    df = pd.DataFrame(data)

    res = normalize_to_instant_kwh(
        df, category="instant_energy", unit="Wh", target_interval_min=15
    )

    # 10:00 to 10:15 span
    assert res["y"].sum() == pytest.approx(0.3)  # 10:15 is the start of next bin
