"""Holdout selection between the ML model and transparent benchmarks.

Every candidate forecasts the same calibration holdout from the same origin
and with the same (historical) weather, so their errors are comparable. The
lowest RMSE wins: for on/off loads such as a heat pump, MAE rewards an
always-off forecast, while RMSE penalises missed runs and favours unbiased
energy, which is what an energy planner integrates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

ML_CANDIDATE = "kpower_ml"
DEGREE_HOUR_CANDIDATE = "degree_hour_regression"
SELECTION_METRIC = "rmse"
# Relative RMSE margin a benchmark must beat ML by. Ties keep the ML model.
SELECTION_MARGIN = 0.0


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    """Holdout errors of one candidate, in target units per interval."""

    rows: int
    rmse: float
    mae: float
    bias: float

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable metric values."""
        return asdict(self)


def score_candidate(
    actual: pd.Series, predicted: pd.Series
) -> Optional[CandidateMetrics]:
    """Score one candidate on rows where both values are finite.

    Args:
        actual: Measured target values (NaN for unmeasured rows).
        predicted: Candidate values on the same index.

    Returns:
        Metrics, or ``None`` when no row can be scored.
    """
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not mask.any():
        return None
    errors = predicted_values[mask] - actual_values[mask]
    return CandidateMetrics(
        rows=int(mask.sum()),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        bias=float(np.mean(errors)),
    )


def select_candidate(metrics: dict[str, CandidateMetrics]) -> str:
    """Return the candidate with the lowest RMSE; ties keep the ML model.

    Args:
        metrics: Scored candidates. Must contain :data:`ML_CANDIDATE`.

    Returns:
        Winning candidate name.
    """
    winner = ML_CANDIDATE
    best = metrics[ML_CANDIDATE].rmse * (1.0 - SELECTION_MARGIN)
    for name, candidate in sorted(metrics.items()):
        if name != ML_CANDIDATE and candidate.rmse < best:
            winner, best = name, candidate.rmse
    return winner


@dataclass(slots=True)
class DegreeHourRegression:
    """Ridge regression on degree hours with per-local-hour intercepts.

    ``y ≈ a[local hour] + b · max(0, base − T_out) + Σ c_i · x_i`` where
    ``x_i`` are optional extra columns (shortwave radiation by default). The
    hour intercepts are effectively unpenalised; the slopes carry a ridge so
    a season without heating demand shrinks them toward zero instead of
    fitting noise. Predictions are clipped at zero.
    """

    base_temperature_c: float
    extra_features: tuple[str, ...]
    timezone: str
    ridge: float = 1.0
    coefficients: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)

    def _design(self, frame: pd.DataFrame, scales: list[float]) -> np.ndarray:
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["ds"], utc=True))
        hours = timestamps.tz_convert(self.timezone).hour.to_numpy()
        temperature = pd.to_numeric(frame["temperature_2m"], errors="coerce")
        columns = [
            np.eye(24)[hours],
            np.clip(
                self.base_temperature_c - temperature.to_numpy(dtype=float), 0, None
            )[:, None],
        ]
        for name, scale in zip(self.extra_features, scales[1:], strict=True):
            values = (
                pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
                if name in frame
                else pd.Series(0.0, index=frame.index)
            )
            columns.append((values.to_numpy(dtype=float) / scale)[:, None])
        return np.hstack(columns)

    def fit(self, frame: pd.DataFrame) -> bool:
        """Fit on rows with a finite target and outdoor temperature.

        Args:
            frame: Rows with ``ds``, ``y``, ``temperature_2m`` and extras.

        Returns:
            Whether a fit was produced (at least one day of usable rows).
        """
        if "temperature_2m" not in frame or frame.empty:
            return False
        usable = frame.loc[
            pd.to_numeric(frame["y"], errors="coerce").notna()
            & pd.to_numeric(frame["temperature_2m"], errors="coerce").notna()
        ]
        if len(usable) < 24:
            return False
        scales = [1.0]
        for name in self.extra_features:
            values = (
                pd.to_numeric(usable[name], errors="coerce")
                if name in usable
                else pd.Series(dtype=float)
            )
            spread = float(values.std()) if values.notna().sum() > 1 else 0.0
            scales.append(spread if np.isfinite(spread) and spread > 0 else 1.0)
        x = self._design(usable, scales)
        y = pd.to_numeric(usable["y"]).to_numpy(dtype=float)
        # A negligible ridge on the hour intercepts keeps the system solvable
        # when a local hour has no observation.
        penalty = np.full(x.shape[1], 1e-6)
        penalty[24:] = self.ridge
        try:
            coefficients = np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        if not np.all(np.isfinite(coefficients)):
            return False
        self.coefficients = coefficients.tolist()
        self.scales = scales
        return True

    @property
    def fitted(self) -> bool:
        """Return whether coefficients are available."""
        return bool(self.coefficients)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict non-negative values for ``frame`` rows.

        Rows without outdoor temperature are NaN.
        """
        if not self.fitted:
            raise ValueError("degree-hour regression is not fitted")
        values = self._design(frame, self.scales) @ np.asarray(self.coefficients)
        missing = pd.to_numeric(frame["temperature_2m"], errors="coerce").isna()
        result = pd.Series(np.clip(values, 0.0, None), index=frame.index)
        return result.mask(missing.to_numpy())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable state."""
        return {
            "base_temperature_c": self.base_temperature_c,
            "extra_features": list(self.extra_features),
            "timezone": self.timezone,
            "ridge": self.ridge,
            "coefficients": self.coefficients,
            "scales": self.scales,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DegreeHourRegression":
        """Restore a regression from :meth:`to_dict` output."""
        return cls(
            base_temperature_c=float(payload["base_temperature_c"]),
            extra_features=tuple(payload.get("extra_features", ())),
            timezone=str(payload["timezone"]),
            ridge=float(payload.get("ridge", 1.0)),
            coefficients=[float(value) for value in payload["coefficients"]],
            scales=[float(value) for value in payload["scales"]],
        )
