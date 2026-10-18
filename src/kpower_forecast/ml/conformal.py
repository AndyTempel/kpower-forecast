"""Split conformal interval calibration for ML forecasts.

Intervals come from signed calibration residuals (``actual - predicted``), so
a band can be wider above the forecast than below it. Residuals are binned by
the predicted value: solar error grows with output and load error with load,
so one global width is too wide at low output and too narrow at high output.
Zero forecasts (solar at night) form their own bin and the rest are split at
quantiles of the positive forecasts, so night rows cannot collapse the bins.
A bin with too few calibration rows uses the global band.
"""

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INTERVAL_BINS = 4
# One day of 15-minute rows: fewer leave a 90 % tail to a handful of points.
DEFAULT_MIN_BIN_SAMPLES = 96

Band = tuple[float, float]


class SplitConformalCalibrator:
    """Calibrate prediction intervals from signed, predicted-value-binned residuals.

    For each level ``L`` the band around a forecast ``yhat`` in bin ``b`` is
    ``[yhat + q_b(a / 2), yhat + q_b(1 - a / 2)]`` with ``a = 1 - L / 100`` and
    ``q_b`` the residual quantile in that bin. The band always contains
    ``yhat`` and its lower edge is never negative.
    """

    def __init__(
        self,
        interval_levels: list[int],
        bins: int = DEFAULT_INTERVAL_BINS,
        min_bin_samples: int = DEFAULT_MIN_BIN_SAMPLES,
    ):
        if bins < 1:
            raise ValueError("bins must be at least 1")
        if min_bin_samples < 1:
            raise ValueError("min_bin_samples must be at least 1")
        self.interval_levels = sorted(set(interval_levels))
        self.bins = bins
        self.min_bin_samples = min_bin_samples
        # Upper predicted-value edges between bins (ascending, unique).
        self.edges: list[float] = []
        # Per level: residual offsets for each bin, then the global offsets.
        self.bands: dict[int, list[Band]] = {}
        self.global_bands: dict[int, Band] = {}

    def fit(self, actual: pd.Series, predicted: pd.Series) -> None:
        """Fit residual quantiles per level and predicted-value bin.

        Args:
            actual: Observed target values.
            predicted: Predicted target values aligned with ``actual``.

        Returns:
            None.

        Raises:
            ValueError: If no residuals are available.
        """
        frame = pd.DataFrame(
            {
                "actual": pd.to_numeric(actual.reset_index(drop=True), errors="coerce"),
                "predicted": pd.to_numeric(
                    predicted.reset_index(drop=True), errors="coerce"
                ),
            }
        ).dropna()
        if frame.empty:
            raise ValueError("cannot calibrate intervals without residuals")

        residuals = frame["actual"] - frame["predicted"]
        positive = frame["predicted"][frame["predicted"] > 0.0]
        cuts = [step / self.bins for step in range(1, self.bins)]
        edges = (
            {float(edge) for edge in positive.quantile(cuts)}
            if len(positive)
            else set()
        )
        if len(positive) < len(frame):
            edges.add(0.0)
        self.edges = sorted(edges)
        bin_index = self._bin_index(frame["predicted"].to_numpy())
        self.bands = {}
        self.global_bands = {}
        for level in self.interval_levels:
            global_band = _residual_band(residuals, level)
            self.global_bands[level] = global_band
            level_bands: list[Band] = []
            for index in range(len(self.edges) + 1):
                in_bin = residuals[bin_index == index]
                level_bands.append(
                    _residual_band(in_bin, level)
                    if len(in_bin) >= self.min_bin_samples
                    else global_band
                )
            self.bands[level] = level_bands

    def apply(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Add calibrated interval columns to a forecast dataframe.

        Args:
            forecast: Forecast with a ``yhat`` column.

        Returns:
            A copy with ``yhat_lower_<L>`` and ``yhat_upper_<L>`` per level.

        Raises:
            ValueError: If ``forecast`` has no ``yhat`` column.
        """
        if "yhat" not in forecast.columns:
            raise ValueError("forecast must contain a 'yhat' column")

        output = forecast.copy()
        yhat = pd.to_numeric(output["yhat"], errors="coerce").to_numpy(dtype=float)
        bin_index = self._bin_index(yhat)
        for level, level_bands in self.bands.items():
            offsets = np.array(level_bands, dtype=float)[bin_index]
            lower = np.minimum(yhat + offsets[:, 0], yhat)
            upper = np.maximum(yhat + offsets[:, 1], yhat)
            output[f"yhat_lower_{level}"] = np.clip(lower, 0.0, None)
            output[f"yhat_upper_{level}"] = upper
        return output

    def to_dict(self) -> dict[str, Any]:
        """Serialize the calibration to a JSON-compatible dictionary.

        Returns:
            The bin settings, edges and per-level bands.
        """
        return {
            "bins": self.bins,
            "min_bin_samples": self.min_bin_samples,
            "edges": list(self.edges),
            "levels": {
                str(level): {
                    "bands": [list(band) for band in self.bands[level]],
                    "global": list(self.global_bands[level]),
                }
                for level in self.bands
            },
        }

    @classmethod
    def from_dict(
        cls, interval_levels: list[int], payload: dict[str, Any]
    ) -> "SplitConformalCalibrator":
        """Load a calibrator from a JSON-compatible dictionary.

        Args:
            interval_levels: Configured interval levels.
            payload: Output of :meth:`to_dict`.

        Returns:
            The restored calibrator.

        Raises:
            ValueError: If the payload is malformed: wrong types, edges that
                are not finite and strictly ascending, or a configured level
                without one band per bin.
        """
        try:
            return cls._from_payload(interval_levels, payload)
        except (TypeError, AttributeError) as exc:
            # A manifest only checks that the payload is a dict; any nested
            # shape error means the stored state is unusable, like any other.
            raise ValueError(f"conformal state is malformed: {exc}") from exc

    @classmethod
    def _from_payload(
        cls, interval_levels: list[int], payload: dict[str, Any]
    ) -> "SplitConformalCalibrator":
        calibrator = cls(
            interval_levels=interval_levels,
            bins=int(payload.get("bins", DEFAULT_INTERVAL_BINS)),
            min_bin_samples=int(
                payload.get("min_bin_samples", DEFAULT_MIN_BIN_SAMPLES)
            ),
        )
        calibrator.edges = [float(edge) for edge in payload.get("edges", [])]
        if not all(np.isfinite(edge) for edge in calibrator.edges) or any(
            later <= earlier for earlier, later in pairwise(calibrator.edges)
        ):
            raise ValueError("conformal edges must be finite and ascending")
        levels = payload.get("levels", {})
        for level in calibrator.interval_levels:
            entry = levels.get(str(level))
            if not isinstance(entry, dict):
                raise ValueError(f"conformal state has no level {level}")
            bands = [_band(item) for item in entry.get("bands", [])]
            if len(bands) != len(calibrator.edges) + 1:
                raise ValueError(f"conformal level {level} has a wrong bin count")
            calibrator.bands[level] = bands
            calibrator.global_bands[level] = _band(entry.get("global"))
        return calibrator

    def _bin_index(self, values: np.ndarray) -> np.ndarray:
        """Bin of each predicted value: values up to an edge fall below it."""
        return np.searchsorted(np.array(self.edges, dtype=float), values, side="left")


def _residual_band(residuals: pd.Series, level: int) -> Band:
    alpha = 1.0 - (level / 100.0)
    return (
        float(residuals.quantile(alpha / 2.0)),
        float(residuals.quantile(1.0 - alpha / 2.0)),
    )


def _band(item: Any) -> Band:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        raise ValueError("a conformal band must be a [lower, upper] pair")
    lower, upper = float(item[0]), float(item[1])
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower > upper:
        raise ValueError("a conformal band must be finite and ordered")
    return lower, upper
