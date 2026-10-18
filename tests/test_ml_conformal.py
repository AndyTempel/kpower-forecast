import numpy as np
import pandas as pd
import pytest

from kpower_forecast.ml.conformal import SplitConformalCalibrator


def _heteroscedastic(seed: int, rows: int = 4000) -> tuple[pd.Series, pd.Series]:
    """Error that grows with output, like PV: 5 % of the predicted value."""
    rng = np.random.default_rng(seed)
    predicted = rng.uniform(0.0, 5.0, rows)
    actual = predicted + rng.normal(0.0, 0.05 * predicted + 0.01)
    return pd.Series(actual), pd.Series(predicted)


def _coverage(result: pd.DataFrame, actual: pd.Series, level: int) -> float:
    inside = (actual >= result[f"yhat_lower_{level}"]) & (
        actual <= result[f"yhat_upper_{level}"]
    )
    return float(inside.mean())


def test_bands_follow_predicted_output_and_hold_coverage_per_bin() -> None:
    calibrator = SplitConformalCalibrator(interval_levels=[90])
    calibrator.fit(*_heteroscedastic(seed=1))
    actual, predicted = _heteroscedastic(seed=2)

    result = calibrator.apply(pd.DataFrame({"yhat": predicted}))

    low = predicted < 1.25
    high = predicted > 3.75
    width = result["yhat_upper_90"] - result["yhat_lower_90"]
    # One global width would be too wide at low output and too narrow at high.
    assert width[low].mean() < width[high].mean() / 3
    assert _coverage(result[low], actual[low], 90) == pytest.approx(0.9, abs=0.03)
    assert _coverage(result[high], actual[high], 90) == pytest.approx(0.9, abs=0.03)


def test_night_zeros_get_their_own_bin() -> None:
    actual, predicted = _heteroscedastic(seed=5)
    night = pd.Series(np.zeros(len(actual)))
    calibrator = SplitConformalCalibrator(interval_levels=[90])
    calibrator.fit(
        pd.concat([night, actual], ignore_index=True),
        pd.concat([night, predicted], ignore_index=True),
    )

    result = calibrator.apply(pd.DataFrame({"yhat": [0.0, 0.3, 4.5]}))

    # Half the rows are zero; the daytime forecasts still split into quartiles.
    assert len(calibrator.edges) == 4
    width = result["yhat_upper_90"] - result["yhat_lower_90"]
    assert width[0] == pytest.approx(0.0)
    assert width[1] < width[2] / 5


def test_skewed_residuals_widen_only_the_side_they_miss() -> None:
    rng = np.random.default_rng(3)
    predicted = pd.Series(np.full(2000, 2.0))
    # Load spikes above the forecast, never far below it.
    actual = predicted + rng.exponential(0.5, 2000) - 0.1
    calibrator = SplitConformalCalibrator(interval_levels=[90])
    calibrator.fit(actual, predicted)

    result = calibrator.apply(pd.DataFrame({"yhat": [2.0, 0.05]}))

    assert result["yhat_upper_90"][0] - 2.0 > 4 * (2.0 - result["yhat_lower_90"][0])
    # The band holds the forecast and never goes negative.
    assert result["yhat_lower_90"][1] == pytest.approx(0.0)
    assert result["yhat_lower_90"][1] <= 0.05 <= result["yhat_upper_90"][1]


def test_sparse_bins_use_the_global_band() -> None:
    calibrator = SplitConformalCalibrator(interval_levels=[80], min_bin_samples=96)
    calibrator.fit(
        actual=pd.Series([1.0, 2.5, 3.0, 4.5]),
        predicted=pd.Series([1.0, 2.0, 3.0, 4.0]),
    )

    assert calibrator.bands[80] == [calibrator.global_bands[80]] * (
        len(calibrator.edges) + 1
    )


def test_state_round_trips_and_rejects_a_wrong_bin_count() -> None:
    calibrator = SplitConformalCalibrator(interval_levels=[50, 90])
    calibrator.fit(*_heteroscedastic(seed=4, rows=1000))
    forecast = pd.DataFrame({"yhat": [0.5, 2.0, 4.5]})

    restored = SplitConformalCalibrator.from_dict([50, 90], calibrator.to_dict())

    pd.testing.assert_frame_equal(restored.apply(forecast), calibrator.apply(forecast))
    payload = calibrator.to_dict()
    payload["levels"]["90"]["bands"].pop()
    with pytest.raises(ValueError, match="wrong bin count"):
        SplitConformalCalibrator.from_dict([50, 90], payload)
    payload = calibrator.to_dict()
    payload["edges"].reverse()
    with pytest.raises(ValueError, match="ascending"):
        SplitConformalCalibrator.from_dict([50, 90], payload)


def test_split_conformal_requires_residuals() -> None:
    calibrator = SplitConformalCalibrator(interval_levels=[90])

    with pytest.raises(ValueError, match="without residuals"):
        calibrator.fit(actual=pd.Series(dtype=float), predicted=pd.Series(dtype=float))
