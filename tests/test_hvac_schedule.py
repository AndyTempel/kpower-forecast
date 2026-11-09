"""Golden parity against actual EMS projection, with no EMS runtime dependency."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from kpower_forecast.hvac.bundle import (
    _AUTOMATION_OPEN_ENDED_HOLD,
    _FUTURE_SKEW,
    effective_mode_spans,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/hvac_schedule_parity.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_effective_spans_match_the_owning_ems_policy(case):
    args = (
        case["payload"],
        case["device_id"],
        datetime.fromisoformat(case["origin"]),
        case["last_mode"],
        (
            datetime.fromisoformat(case["last_mode_event"])
            if case["last_mode_event"]
            else None
        ),
    )
    end = datetime.fromisoformat(case["horizon_end"])
    if case["error"]:
        with pytest.raises(ValueError, match=case["error"]):
            effective_mode_spans(*args, horizon_end=end)
    else:
        actual = effective_mode_spans(*args, horizon_end=end)
        assert [s.model_dump(mode="json") for s in actual] == case["expected"]
    assert (
        _AUTOMATION_OPEN_ENDED_HOLD.total_seconds()
        == REFERENCE["policy"]["automation_open_ended_hold_s"]
    )
    assert _FUTURE_SKEW.total_seconds() == REFERENCE["policy"]["future_skew_s"]
