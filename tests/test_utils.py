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
    data = {
        "ds": ["2024-01-01 10:00", "2024-01-01 10:15", "2024-01-01 10:30"],
        "y": [100.0, 105.5, 110.0],
    }
    df = pd.DataFrame(data)

    res = normalize_to_instant_kwh(
        df, category="cumulative_energy", unit="kWh", target_interval_min=15
    )

    # Expected diffs: 5.5, 4.5
    assert 5.5 in res["y"].values
    assert 4.5 in res["y"].values


def test_normalize_power_inconsistent_intervals():
    # Power readings (kW) at inconsistent times
    # 10:00: 10kW
    # 10:06: 20kW
    #   Delta = 6 min = 0.1h. Avg Power = (10+20)/2 = 15kW.
    #   Energy = 15 * 0.1 = 1.5kWh.
    # 10:15: 10kW
    #   Delta = 9 min = 0.15h. Avg Power = (20+10)/2 = 15kW.
    #   Energy = 15 * 0.15 = 2.25kWh.
    data = {
        "ds": ["2024-01-01 10:00", "2024-01-01 10:06", "2024-01-01 10:15"],
        "y": [10.0, 20.0, 10.0],
    }
    df = pd.DataFrame(data)

    res = normalize_to_instant_kwh(
        df, category="power", unit="kW", target_interval_min=15
    )

    # Total kWh = 1.5 + 2.25 = 3.75kWh
    assert res["y"].sum() == pytest.approx(3.75)


def test_normalize_instant_energy_resampling():
    # Instant readings (Wh) every 5 minutes
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

    # 100Wh = 0.1kWh.
    # Resampled 15min: 10:00 to 10:14 should be summed (0.3kWh).
    # 10:15 goes to next bucket (0.1kWh).
    assert any(val == pytest.approx(0.3) for val in res["y"].values)
    assert any(val == pytest.approx(0.1) for val in res["y"].values)
