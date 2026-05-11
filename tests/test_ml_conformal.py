import pandas as pd
import pytest

from kpower_forecast.ml.conformal import SplitConformalCalibrator


def test_split_conformal_adds_interval_columns() -> None:
    calibrator = SplitConformalCalibrator(interval_levels=[50, 90])
    calibrator.fit(
        actual=pd.Series([1.0, 2.0, 3.0]),
        predicted=pd.Series([1.0, 1.5, 2.0]),
    )
    forecast = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=2), "yhat": [1.0, 2.0]}
    )

    result = calibrator.apply(forecast)

    assert {
        "yhat_lower_50",
        "yhat_upper_50",
        "yhat_lower_90",
        "yhat_upper_90",
    }.issubset(result.columns)
    assert result["yhat_lower_90"].min() >= 0.0


def test_split_conformal_requires_residuals() -> None:
    calibrator = SplitConformalCalibrator(interval_levels=[90])

    with pytest.raises(ValueError, match="without residuals"):
        calibrator.fit(actual=pd.Series(dtype=float), predicted=pd.Series(dtype=float))
