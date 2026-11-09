"""Directional settlement, feasible peak allocation and full-ensemble reductions."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kpower_forecast.hvac.settlement import SettlementWindow, settlement_need
from kpower_forecast.hvac.types import MinutePaths

START = datetime(2026, 11, 9, tzinfo=UTC)


def _need(values, allowance, hard=None, *, start=START, end=None):
    values = np.atleast_2d(values).astype(np.float32)
    paths = MinutePaths(START, values, None)
    window = SettlementWindow(
        start, end or START + timedelta(minutes=values.shape[1]), "a", allowance, hard
    )
    return settlement_need(
        paths, np.zeros(values.shape[1]), np.zeros(values.shape[1]), [window]
    )


@pytest.mark.parametrize(
    "allowance,limit,energy,power",
    [
        (None, 600.0, 100.0, 400.0),
        (100.0, None, 150.0, 600.0),
        (100.0, 600.0, 150.0, 600.0),
        (1000.0, 0.0, 250.0, 1000.0),
        (None, None, 0.0, 0.0),
        (-50.0, None, 250.0, 1000.0),
    ],
)
def test_hard_shave_precedes_average_and_past_overrun_is_not_future_need(
    allowance, limit, energy, power
):
    result = _need(np.full(15, 1000.0), allowance, limit)
    block = result.blocks[0]
    assert block.energy_q_wh == pytest.approx([energy, energy])
    assert block.power_q_w[:, 0] == pytest.approx([power, power])
    assert block.past_overrun_wh == (50.0 if allowance == -50.0 else 0.0)
    assert result.windows[0].import_allowance_wh == allowance


def test_export_never_refunds_import_and_current_allowance_is_already_remaining():
    paths = MinutePaths(START, np.full((1, 15), 1000.0, np.float32), None)
    pv = np.array([0.0, 2000.0] * 7 + [0.0])
    result = settlement_need(
        paths,
        np.zeros(15),
        pv,
        [SettlementWindow(START, START + timedelta(minutes=15), "a", 100.0, None)],
    )
    assert result.blocks[0].energy_q_wh[0] == pytest.approx(8 * 1000 / 60 - 100)
    assert max((paths.power_w[0] - pv).sum() / 60 - 100, 0) == 0
    partial = _need(np.full(5, 1000.0), 30.0, start=START - timedelta(minutes=10))
    assert partial.blocks[0].energy_q_wh[0] == pytest.approx(5000 / 60 - 30)


def test_capped_allocation_handles_low_import_and_fractional_minute_boundaries():
    # A uniform505W allocation exports495W in minute0 and leaves495W in minute1.
    result = _need([10.0, 1000.0], 0.0)
    assert result.blocks[0].energy_q_wh[0] == pytest.approx(1010 / 60)
    assert result.blocks[0].power_q_w[0, 0] == pytest.approx(1000.0, abs=1e-9)
    fractional = _need(
        [10.0, 1000.0],
        0.0,
        start=START + timedelta(seconds=15),
        end=START + timedelta(minutes=1, seconds=45),
    )
    assert fractional.blocks[0].energy_q_wh[0] == pytest.approx(1010 * 0.75 / 60)
    assert fractional.blocks[0].power_q_w[0, 0] == pytest.approx(1000.0, abs=1e-9)


def test_block_quantiles_are_after_path_sums_and_nonconsecutive_ids_stay_separate():
    values = np.zeros((256, 45), np.float32)
    values[:128, :15] = 1000.0
    values[128:, 15:30] = 1000.0
    paths = MinutePaths(START, values, None)
    windows = [
        SettlementWindow(
            START + timedelta(minutes=i),
            START + timedelta(minutes=i + 15),
            "a" if i < 30 else "b",
            0.0,
            None,
        )
        for i in (0, 15, 30)
    ]
    result = settlement_need(paths, np.zeros(45), np.zeros(45), windows)
    assert result.blocks[0].window_indices == (0, 1)
    assert result.blocks[0].energy_q_wh[1] == 250.0  # not250+250 marginal q90s
    assert result.blocks[0].paths_needing == 1.0
    assert not result.blocks[0].per_path_energy_wh.flags.writeable
    separate = [
        SettlementWindow(w.start, w.end, "b" if i == 1 else "a", 0.0, None)
        for i, w in enumerate(windows)
    ]
    assert len(settlement_need(paths, np.zeros(45), np.zeros(45), separate).blocks) == 3


def test_hand_built_ensemble_quantiles_and_invalid_inputs():
    levels = np.arange(256, dtype=float) * 10 + 500
    result = _need(np.repeat(levels[:, None], 15, axis=1), 100.0)
    expected = np.maximum(levels / 4 - 100, 0)
    np.testing.assert_allclose(
        result.blocks[0].energy_q_wh, np.quantile(expected, [0.5, 0.9])
    )
    assert result.blocks[0].paths_needing == np.mean(expected > 0)
    paths = MinutePaths(START, np.ones((16, 15), np.float32), None)
    windows = [SettlementWindow(START, START + timedelta(minutes=15), "a", 0.0, None)]
    with pytest.raises(ValueError, match="finite"):
        settlement_need(paths, np.full(15, np.nan), np.zeros(15), windows)
    with pytest.raises(ValueError, match="quantiles"):
        settlement_need(paths, np.zeros(15), np.zeros(15), windows, [0.9, 0.5])
    with pytest.raises(ValueError, match="nonoverlapping"):
        settlement_need(paths, np.zeros(15), np.zeros(15), windows * 2)
    with pytest.raises(ValueError, match="negative"):
        SettlementWindow(START, START + timedelta(minutes=15), "a", 0.0, -1.0)
