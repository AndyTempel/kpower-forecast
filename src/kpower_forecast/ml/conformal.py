"""Split conformal interval calibration for ML forecasts."""

import pandas as pd


class SplitConformalCalibrator:
    """Calibrate symmetric prediction intervals from absolute residuals."""

    def __init__(self, interval_levels: list[int]):
        self.interval_levels = sorted(set(interval_levels))
        self.quantiles: dict[int, float] = {}

    def fit(self, actual: pd.Series, predicted: pd.Series) -> None:
        """Fit non-conformity quantiles.

        Args:
            actual: Observed target values.
            predicted: Predicted target values aligned with ``actual``.

        Returns:
            None.

        Raises:
            ValueError: If no residuals are available.
        """
        residuals = (
            actual.reset_index(drop=True) - predicted.reset_index(drop=True)
        ).abs()
        residuals = pd.to_numeric(residuals, errors="coerce").dropna()
        if residuals.empty:
            raise ValueError("cannot calibrate intervals without residuals")

        for level in self.interval_levels:
            alpha = 1.0 - (level / 100.0)
            self.quantiles[level] = float(residuals.quantile(1.0 - alpha))

    def apply(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Add calibrated interval columns to a forecast dataframe."""
        if "yhat" not in forecast.columns:
            raise ValueError("forecast must contain a 'yhat' column")

        output = forecast.copy()
        for level, width in self.quantiles.items():
            output[f"yhat_lower_{level}"] = (output["yhat"] - width).clip(lower=0.0)
            output[f"yhat_upper_{level}"] = output["yhat"] + width
        return output

    def to_dict(self) -> dict[str, float]:
        """Serialize quantiles to a JSON-compatible dictionary."""
        return {str(level): width for level, width in self.quantiles.items()}

    @classmethod
    def from_dict(
        cls, interval_levels: list[int], payload: dict[str, float]
    ) -> "SplitConformalCalibrator":
        """Load a calibrator from a JSON-compatible dictionary."""
        calibrator = cls(interval_levels=interval_levels)
        calibrator.quantiles = {
            int(level): float(width) for level, width in payload.items()
        }
        return calibrator
