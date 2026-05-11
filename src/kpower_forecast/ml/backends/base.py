"""Backend protocol for optional ML forecasting engines."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ForecastBackend(Protocol):
    """Protocol implemented by ML forecasting backends."""

    def fit(
        self,
        history: pd.DataFrame,
        features: pd.DataFrame,
        calibration: pd.DataFrame,
    ) -> None:
        """Fit the backend to historical observations and features."""

    def predict(self, future_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Generate a point forecast for the requested future horizon."""

    def save(self, path: Path) -> dict[str, str]:
        """Persist backend artifacts and return manifest-relative paths."""

    def load(self, path: Path) -> None:
        """Load backend artifacts from ``path``."""

    def feature_schema(self) -> list[str]:
        """Return the feature columns expected at inference time."""
