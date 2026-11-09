"""Input identity and full-ensemble boundaries remain explicit."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from kpower_forecast.hvac.types import (
    GridInterval,
    HvacCapabilities,
    HvacForecastRequest,
    HvacModelConfig,
    MinutePaths,
    ModeSpan,
)


@pytest.fixture
def origin():
    return datetime(2026, 11, 9, tzinfo=timezone.utc)


@pytest.fixture
def request_inputs(origin):
    return dict(
        origin=origin,
        recent=pd.DataFrame(
            {"power_w": [0.0, np.nan]},
            index=pd.date_range(origin, periods=2, freq="min"),
        ),
        schedule=(),
        last_mode=None,
        weather_forecast=pd.DataFrame(),
        grid=(GridInterval(start=origin, minutes=15),),
    )


def test_request_preserves_gaps_and_unknown_modes_without_mutating_caller(
    request_inputs,
):
    request = HvacForecastRequest(**request_inputs)
    request_inputs["recent"].iloc[0, 0] = 9000
    assert request.recent.power_w.iloc[0] == 0
    assert pd.isna(request.recent.power_w.iloc[1]) and request.last_mode is None


@pytest.mark.parametrize("damage", ["naive", "duplicate", "overlap", "deadline"])
def test_invalid_time_grid_or_budget_is_rejected(request_inputs, origin, damage):
    if damage == "naive":
        request_inputs["origin"] = origin.replace(tzinfo=None)
    elif damage == "duplicate":
        request_inputs["recent"].index = [origin, origin]
    elif damage == "overlap":
        request_inputs["schedule"] = tuple(
            ModeSpan(
                start=origin,
                end=origin + pd.Timedelta(hours=1),
                mode="ECO",
                source="plan",
            )
            for _ in range(2)
        )
    else:
        request_inputs["deadline_s"] = float("inf")
    with pytest.raises(ValueError):
        HvacForecastRequest(**request_inputs)


def test_family_contract_accepts_inverter_without_inventing_stages_or_rated_points():
    caps = HvacCapabilities(family="inverter", fingerprint="unit")
    assert caps.stages is None and caps.rated_points == ()
    with pytest.raises(ValidationError):
        HvacModelConfig(model_id="site", timezone="UTC", reference_mode="OFF")


def test_minute_paths_copy_ensemble_and_keep_energy_only_state_unknown(origin):
    power = np.full((256, 60), 1000.0, dtype=np.float32)
    paths = MinutePaths(origin, power, None)
    power[:] = 99
    assert np.all(paths.power_w == 1000) and paths.running is None
    assert not paths.power_w.flags.writeable
    with pytest.raises(ValueError):
        MinutePaths(origin, power, np.zeros((16, 60), dtype=bool))
