"""Censored, recency-weighted duration fits recover the generating process."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from kpower_forecast.hvac.cycle_model import CycleFit, fit_cycles
from kpower_forecast.hvac.types import HvacModelConfig

AS_OF = datetime(2026, 11, 9, tzinfo=UTC)


def spells(n=600, *, idle_slope=0.06, censor=0.2, seed=8):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp(AS_OF) - pd.Timedelta(days=10)
    for i in range(n):
        t = rng.uniform(-2, 14)
        mode = "ECO" if i % 3 == 0 else "NORMAL"
        run = 18 * np.exp(-0.03 * (t - 7) + 0.25 * rng.normal())
        prev = run
        idle = (
            45
            * np.exp(idle_slope * (t - 7) + 0.3 * (mode == "ECO"))
            * (prev / 18) ** 0.5
            * np.exp(0.25 * rng.normal())
        )
        for kind, minutes, p in (("run", run, np.nan), ("idle", idle, prev)):
            cut = rng.random() < censor
            rows.append(
                dict(
                    kind=kind,
                    start=start + pd.Timedelta(minutes=30 * i),
                    minutes=minutes * (rng.uniform(0.2, 1) if cut else 1),
                    end_cause="command" if cut else "natural",
                    censored=cut,
                    mode=mode,
                    outdoor_c=t,
                    sun=0.0,
                    prev_run_min=p,
                )
            )
    return pd.DataFrame(rows)


def test_fit_recovers_slopes_mode_effect_previous_run_and_dispersion():
    cfg = HvacModelConfig(model_id="t", timezone="UTC", recency_half_life_days=1000)
    fit, reason, flags = fit_cycles(spells(), cfg, as_of=AS_OF)
    assert reason is None and not flags
    assert fit.modes == ("ECO",)
    assert fit.idle_beta[1] == pytest.approx(0.06, abs=0.01)
    assert fit.idle_beta[3] == pytest.approx(0.3, abs=0.06)
    assert fit.idle_beta[4] == pytest.approx(0.5, abs=0.12)
    assert fit.run_beta[1] == pytest.approx(-0.03, abs=0.01)
    assert fit.idle_sigma == pytest.approx(0.25, abs=0.03)
    assert fit.run_sigma == pytest.approx(0.25, abs=0.03)
    duty, idle, run = fit.stationary(7.0, "NORMAL")
    assert idle == pytest.approx(45, rel=0.1) and run == pytest.approx(18, rel=0.1)


def test_wrong_sign_temperature_slope_is_dropped_and_thin_history_falls_back():
    cfg = HvacModelConfig(model_id="t", timezone="UTC")
    fit, _, flags = fit_cycles(spells(idle_slope=-0.05), cfg, as_of=AS_OF)
    assert flags["idle_temperature_slope_dropped"] and fit.idle_beta[1] == 0.0
    fit, reason, _ = fit_cycles(spells(n=10), cfg, as_of=AS_OF)
    assert fit is None and reason == "insufficient_cycles"
    narrow = spells()
    narrow["outdoor_c"] = 5.0
    assert fit_cycles(narrow, cfg, as_of=AS_OF)[1] == "narrow_outdoor_range"


def test_invalid_coefficients_are_rejected():
    with pytest.raises(ValueError):
        CycleFit((), (1.0, 0.0, 0.0), 0.2, (1.0, 0.0, 0.0), 0.2, 15, 0, 10, 1, 1)
