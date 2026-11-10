"""Run-power ramp regression and measured standby."""

import numpy as np
import pandas as pd

from kpower_forecast.hvac.power import fit_power
from kpower_forecast.hvac.types import HvacModelConfig, MeteredScope


def test_power_recovers_rising_run_level_ramp_and_standby():
    rng = np.random.default_rng(4)
    rows = []
    for cycle in range(40):
        t = 10.0 - cycle * 0.25
        rows += [(0, 95.0, t)] * 40
        for k in range(20):
            level = 4000 + 60 * min(k, 30) - 30 * (t - 7)
            factor = 0.6 if k == 0 else 1.0
            rows.append((1, level * factor + rng.normal(0, 20), t))
    running, power, outdoor = map(np.array, zip(*rows, strict=True))
    power[50] = 20000.0  # an outlier must not move the Huber fit
    index = pd.date_range("2026-11-01", periods=len(rows), freq="min", tz="UTC")
    frame = pd.DataFrame(
        {"running": running.astype(float), "power_w": power, "outdoor_c": outdoor},
        index=index,
    )
    events = pd.DataFrame({"time": [index[0]], "mode": ["NORMAL"], "source": "plan"})
    cfg = HvacModelConfig(model_id="t", timezone="UTC")
    fit = fit_power(frame, events, cfg, MeteredScope.WHOLE_UNIT)
    assert abs(fit.per_c_w + 30) < 5
    # The stop minute is modelled separately and does not lower the late profile.
    assert len(fit.profile_w) == 19
    assert fit.stop_w is not None and abs(fit.stop_w - (4000 + 60 * 19)) < 60
    assert abs(fit.profile_w[0] - 0.6 * 4000) < 60  # partial first minute
    assert abs(fit.profile_w[10] - 4600) < 60  # one 20 kW spike is clipped
    assert fit.idle_w == 95.0 and fit.off_w == 95.0
    p = fit.run_power(np.array([2, 10, 19, 60]), 7.0)
    assert np.all(np.diff(p[:3]) > 0) and p[3] == p[2]  # holds past the profile
