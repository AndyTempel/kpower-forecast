"""Cold extrapolation of a warm-trained cycle model keeps physical shape."""

import numpy as np

from kpower_forecast.hvac.cycle_model import CycleFit


def test_colder_means_more_duty_shorter_idles_longer_runs():
    fit = CycleFit(
        ("ECO",),
        (2.0, 0.05, 0.1, 0.3, 0.6),
        0.3,
        (3.2, -0.04, -0.1, 0.0),
        0.25,
        20.0,
        2.0,
        15.0,
        100,
        100,
    )
    temps = [15.0, 7.0, 0.0, -10.0, -20.0]
    rows = [fit.stationary(t, "NORMAL") for t in temps]
    duty, idle, run = map(np.array, zip(*rows, strict=True))
    assert np.all(np.diff(duty) > 0) and np.all(duty < 1)
    assert np.all(np.diff(idle) < 0) and np.all(np.diff(run) > 0)
    # A mode with a fitted longer idle lowers duty at the same temperature.
    assert fit.stationary(0.0, "ECO")[0] < fit.stationary(0.0, "NORMAL")[0]
