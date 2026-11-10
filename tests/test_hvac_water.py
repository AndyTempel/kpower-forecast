"""Buffer-temperature nowcast of the current idle."""

import numpy as np
import pandas as pd

from kpower_forecast.hvac.water import current_water, fit_water, water_estimates


def _idles(lengths, rate=0.2, setpoint=30.0, hysteresis=4.0):
    """Linear cooling from 8 K above the restart threshold, then a 20-min run."""
    control, spells, t = [], [], pd.Timestamp("2026-11-08", tz="UTC")
    start = t
    for length in lengths:
        cooling = np.linspace(8.0, 8.0 - rate * (length - 1), length)
        control += list(setpoint - hysteresis + cooling) + [setpoint] * 20
        spells.append(("idle", start, length, False))
        start += pd.Timedelta(minutes=length + 20)
    index = pd.date_range(t, periods=len(control), freq="min")
    frame = pd.DataFrame(
        {
            "control_c": control,
            "control_setpoint_c": setpoint,
            "control_hysteresis_c": hysteresis,
        },
        index=index,
    )
    rows = pd.DataFrame(spells, columns=["kind", "start", "minutes", "censored"])
    return frame, rows


def test_remaining_idle_is_gap_over_cooling_rate_and_calibrates_on_complete_idles():
    frame, spells = _idles([41] * 8)
    estimates = water_estimates(frame)
    # After 20 idle minutes 4.2 K are left at 0.2 K/min.
    assert abs(estimates.iloc[19] - 21.0) < 0.5
    fit = fit_water(frame, spells)
    assert fit is not None and abs(fit.bias) < 0.05
    origin = frame.index[19] + pd.Timedelta(minutes=3)
    assert abs(current_water(frame.iloc[:20], origin) - 19.0) < 0.5
    # Too few idles with buffer evidence: no nowcast instead of a guess.
    assert fit_water(*_idles([41] * 3)) is None
