"""Rolling-origin selection between the ML model and transparent benchmarks.

Every candidate forecasts the same day-ahead windows, each from a model trained
only on history before the window's origin and with the same historical
weather, so their errors are comparable and match how forecasts are used.

Candidates are ranked by RMSE of 1 h means taken at every step (a sliding
window inside each origin's window). A run forecast 15 minutes early then costs
a quarter of a run instead of a missed and a phantom run, and no bin edge
splits a near miss. MAE is not used: it rewards an always-off forecast of an
on/off load. RMSE alone can still prefer a smooth forecast that misses a large
share of the energy, which an energy planner integrates, so a candidate whose
mean error exceeds ``bias_tolerance`` of the mean actual load (or
``bias_floor``) is not eligible. If no candidate is eligible, the least
biased wins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from kpower_forecast.ml.baselines import BASELINE_NAME

ML_CANDIDATE = "kpower_ml"
DEGREE_HOUR_CANDIDATE = "degree_hour_regression"
BLEND_CANDIDATE = "ml_regression_blend"
SELECTION_CANDIDATES: tuple[str, ...] = (
    ML_CANDIDATE,
    DEGREE_HOUR_CANDIDATE,
    BASELINE_NAME,
    BLEND_CANDIDATE,
)
SELECTION_METRIC = "rmse_1h_sliding"


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    """Backtest errors of one candidate, in target units per interval.

    ``rmse`` and ``mae`` are per interval; ``rmse_1h`` uses sliding 1 h means
    and ranks candidates. ``bias`` is the mean signed error and
    ``mean_actual`` the mean measured target over the same rows.
    """

    rows: int
    rmse: float
    mae: float
    bias: float
    rmse_1h: float = float("nan")
    mean_actual: float = float("nan")
    origins: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable metric values."""
        return asdict(self)


def _sliding_mean(values: np.ndarray, steps: int) -> np.ndarray:
    """Mean over every ``steps``-long window; NaN where any value is missing."""
    if steps <= 1:
        return values
    if len(values) < steps:
        return np.array([], dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, steps)
    return np.asarray(windows.mean(axis=1), dtype=float)


def score_windows(
    windows: Sequence[tuple[np.ndarray, np.ndarray]], *, window_steps: int
) -> Optional[CandidateMetrics]:
    """Score a candidate over backtest windows.

    Sliding means never cross from one window into the next.

    Args:
        windows: ``(actual, predicted)`` arrays per origin, NaN where missing.
        window_steps: Intervals per sliding mean (4 for 1 h at 15 minutes).

    Returns:
        Metrics, or ``None`` when no row can be scored.
    """
    errors: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    sliding: list[np.ndarray] = []
    origins = 0
    for actual, predicted in windows:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual) & np.isfinite(predicted)
        if not mask.any():
            continue
        origins += 1
        errors.append(predicted[mask] - actual[mask])
        actuals.append(actual[mask])
        difference = _sliding_mean(predicted, window_steps) - _sliding_mean(
            actual, window_steps
        )
        sliding.append(difference[np.isfinite(difference)])
    if not errors:
        return None
    error = np.concatenate(errors)
    hourly = np.concatenate(sliding)
    return CandidateMetrics(
        rows=int(error.size),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(np.mean(error)),
        # Without a fully measured hour, rank on per-interval RMSE so stored
        # metrics stay finite (they are published as JSON).
        rmse_1h=(
            float(np.sqrt(np.mean(hourly**2)))
            if hourly.size
            else float(np.sqrt(np.mean(error**2)))
        ),
        mean_actual=float(np.mean(np.concatenate(actuals))),
        origins=origins,
    )


def bias_eligible(metrics: CandidateMetrics, *, tolerance: float, floor: float) -> bool:
    """Return whether a candidate's mean error is within the bias guard."""
    limit = max(tolerance * abs(metrics.mean_actual), floor)
    return abs(metrics.bias) <= limit


def _ranking_error(metrics: CandidateMetrics) -> float:
    return metrics.rmse_1h if np.isfinite(metrics.rmse_1h) else metrics.rmse


def select_candidate(
    metrics: dict[str, CandidateMetrics],
    *,
    bias_tolerance: float = float("inf"),
    bias_floor: float = 0.0,
) -> str:
    """Return the winning candidate; ties keep the ML model.

    Among candidates within the bias guard, the lowest sliding 1 h RMSE wins.
    If none is within it, the smallest absolute bias wins.

    Args:
        metrics: Scored candidates. Must contain :data:`ML_CANDIDATE`.
        bias_tolerance: Allowed share of mean actual load.
        bias_floor: Allowed absolute bias in target units per interval.

    Returns:
        Winning candidate name.
    """
    eligible = {
        name: candidate
        for name, candidate in metrics.items()
        if bias_eligible(candidate, tolerance=bias_tolerance, floor=bias_floor)
    }
    if not eligible:
        key = {name: abs(candidate.bias) for name, candidate in metrics.items()}
    else:
        key = {name: _ranking_error(candidate) for name, candidate in eligible.items()}
    winner = ML_CANDIDATE if ML_CANDIDATE in key else min(key, key=key.__getitem__)
    best = key[winner]
    for name in sorted(key):
        if name != winner and key[name] < best:
            winner, best = name, key[name]
    return winner


def fit_blend_weight(
    actual: Sequence[np.ndarray],
    ml: Sequence[np.ndarray],
    regression: Sequence[np.ndarray],
) -> float:
    """Least-squares weight ``w`` of ``w·ml + (1 − w)·regression``, in [0, 1].

    Args:
        actual: Measured values per window.
        ml: ML predictions per window.
        regression: Regression predictions per window.

    Returns:
        The weight; 1.0 (pure ML) when the two forecasts never differ.
    """
    a = np.concatenate([np.asarray(x, dtype=float) for x in actual])
    m = np.concatenate([np.asarray(x, dtype=float) for x in ml])
    r = np.concatenate([np.asarray(x, dtype=float) for x in regression])
    mask = np.isfinite(a) & np.isfinite(m) & np.isfinite(r)
    spread = (m - r)[mask]
    denominator = float(np.dot(spread, spread))
    if denominator <= 0.0:
        return 1.0
    weight = float(np.dot(spread, (a - r)[mask])) / denominator
    return float(np.clip(weight, 0.0, 1.0))


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
            if name not in frame:
                # Zero-filling an input the fit used would serve another model.
                raise ValueError(f"degree-hour regression input {name!r} is missing")
            values = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
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
        # Only inputs with data take part; prediction then requires them.
        self.extra_features = tuple(
            name
            for name in self.extra_features
            if name in usable
            and pd.to_numeric(usable[name], errors="coerce").notna().any()
        )
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
