"""ML weather bias correction primitives."""

from numbers import Real

import pandas as pd


class WeatherBiasCorrector:
    """Small learned correction layer for forecast weather features."""

    def __init__(self, min_samples: int):
        self.min_samples = min_samples
        self.source: str | None = None
        self.irradiance_multiplier: float = 1.0

    def fit_from_archive(self, archive: pd.DataFrame) -> bool:
        """Fit from archived forecast rows matched to actual observations."""
        required = {"y", "shortwave_radiation", "pre_weather_correction_yhat"}
        if len(archive) < self.min_samples or not required.issubset(archive.columns):
            return False

        baseline = archive["pre_weather_correction_yhat"].replace(0.0, pd.NA)
        ratios = pd.to_numeric(archive["y"] / baseline, errors="coerce").dropna()
        if ratios.empty:
            return False

        self.irradiance_multiplier = float(ratios.median())
        self.source = "forecast_archive"
        return True

    def fit_from_historical_proxy(self, history: pd.DataFrame) -> bool:
        """Fit a conservative fallback from historical proxy data."""
        if (
            len(history) < self.min_samples
            or "shortwave_radiation" not in history.columns
        ):
            return False
        self.irradiance_multiplier = 1.0
        self.source = "historical_proxy"
        return True

    def apply(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Apply learned weather corrections to a weather dataframe."""
        corrected = weather.copy()
        for column in ["shortwave_radiation", "direct_radiation", "diffuse_radiation"]:
            if column in corrected.columns:
                corrected[column] = (
                    corrected[column] * self.irradiance_multiplier
                ).clip(lower=0.0)
        return corrected

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Serialize correction state for model manifests."""
        return {
            "min_samples": self.min_samples,
            "source": self.source,
            "irradiance_multiplier": self.irradiance_multiplier,
        }

    def load_dict(self, payload: dict[str, object]) -> None:
        """Load correction state from a model manifest payload."""
        source = payload.get("source")
        self.source = source if isinstance(source, str) else None
        multiplier = payload.get("irradiance_multiplier", 1.0)
        if isinstance(multiplier, Real):
            self.irradiance_multiplier = float(multiplier)
        elif isinstance(multiplier, str):
            self.irradiance_multiplier = float(multiplier)
        else:
            self.irradiance_multiplier = 1.0
