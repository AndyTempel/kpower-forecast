"""Site weather MOS and causal hourly AR(1) outdoor uncertainty.

Weather inputs retain Open-Meteo units: temperature C, radiation W/m², wind km/h.
No weather is fetched here; the caller owns vintages and availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .types import HvacError


@dataclass(frozen=True)
class OutdoorFit:
    """Small weather-to-site mapping and its hourly residual process."""

    coefficients: tuple[float, ...]
    phi: float
    stationary_sd_c: float
    on_reanalysis: bool
    lead_unaware: bool = True
    weather_reference: bool = False

    def __post_init__(self) -> None:
        if (
            len(self.coefficients) != 6
            or not np.isfinite(self.coefficients).all()
            or not 0.5 <= self.phi <= 0.98
            or not np.isfinite(self.stationary_sd_c)
            or self.stationary_sd_c < 0
        ):
            raise HvacError("invalid outdoor calibration")

    def mean(self, weather: pd.DataFrame, timezone: str) -> NDArray[np.float64]:
        """Map an available forecast to the selected site's outdoor series.

        Args:
            weather: Hourly weather with all MOS variables.
            timezone: Site zone used only for local-hour features.

        Returns:
            Site temperatures on the same valid-time grid.
        """
        return np.asarray(_design(weather, timezone) @ np.asarray(self.coefficients))


def _design(weather: pd.DataFrame, timezone: str) -> NDArray[np.float64]:
    required = ["temperature_2m", "shortwave_radiation", "wind_speed_10m"]
    if not set(required) <= set(weather):
        raise HvacError("MOS weather lacks temperature, radiation or wind")
    index = pd.DatetimeIndex(weather.index)
    if (
        str(index.tz) != "UTC"
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise HvacError("MOS weather timestamps require sorted unique UTC")
    local = index.tz_convert(timezone)
    angle = 2 * np.pi * (local.hour + local.minute / 60) / 24
    return np.column_stack(
        [
            np.ones(len(weather)),
            weather.temperature_2m.to_numpy(),
            weather.shortwave_radiation.to_numpy() / 1000,
            weather.wind_speed_10m.to_numpy(),
            np.sin(angle),
            np.cos(angle),
        ]
    )


def fit_outdoor(
    frame: pd.DataFrame, weather: pd.DataFrame, timezone: str
) -> OutdoorFit:
    """Fit on historical forecasts, using reanalysis only below seven forecast days.

    Args:
        frame: Selected measured outdoor and its source labels on UTC minutes.
        weather: UTC source-labelled historical hourly weather (or `ds` column).
        timezone: Site zone for local-hour features.

    Returns:
        Ridge MOS and AR1 residual parameters. Missing fit evidence fails instead
        of treating weather as measured site temperature.

    Raises:
        ValueError: If weather/evidence cannot identify the mapping.
    """
    site = (
        frame.loc[frame.outdoor_source.isin(["unit", "environment"]), "outdoor_c"]
        .resample("h")
        .mean()
        .dropna()
    )
    source = weather.copy().replace([np.inf, -np.inf], np.nan)
    if "ds" in source:
        source = source.set_index("ds")
    source.index = pd.DatetimeIndex(source.index)
    if "weather_source" not in source:
        raise HvacError("MOS fitting requires weather-source provenance")
    forecast = source.loc[source.weather_source.eq("historical_forecast")]
    weather_reference = len(site) < 7
    if weather_reference:
        site = source.loc[
            source.weather_source.eq("reanalysis"), "temperature_2m"
        ].dropna()
        site = site.groupby(level=0).first()
    available = forecast.index.intersection(site.index)
    valid = (
        forecast.loc[
            available, ["temperature_2m", "shortwave_radiation", "wind_speed_10m"]
        ]
        .notna()
        .all(axis=1)
    )
    on_reanalysis = int(valid.sum()) < 7 * 24 and not weather_reference
    if on_reanalysis:
        # Keep archived forecasts first, then fill missing fields from reanalysis;
        # the flag records that reanalysis was admitted to the MOS fit.
        source = source.loc[
            source.weather_source.isin(["historical_forecast", "reanalysis"])
        ].copy()
        source["_priority"] = source.weather_source.map(
            {"historical_forecast": 0, "reanalysis": 1}
        )
        selected = (
            source.sort_values("_priority")
            .groupby(level=0, sort=True)[
                ["temperature_2m", "shortwave_radiation", "wind_speed_10m"]
            ]
            .first()
        )
    else:
        selected = forecast[["temperature_2m", "shortwave_radiation", "wind_speed_10m"]]
    selected = selected.loc[selected.index.intersection(site.index)].dropna()
    if len(selected) < 7:
        raise HvacError("insufficient_weather_calibration")
    x = _design(selected, timezone)
    y = site.reindex(selected.index).to_numpy(dtype=float)
    beta = np.linalg.solve(x.T @ x + np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]), x.T @ y)
    residual = y - x @ beta
    pair = np.asarray(selected.index[1:] - selected.index[:-1] == pd.Timedelta(hours=1))
    denominator = float(np.dot(residual[:-1][pair], residual[:-1][pair]))
    phi = (
        float(np.dot(residual[1:][pair], residual[:-1][pair]) / denominator)
        if denominator > 1e-12
        else 0.5
    )
    return OutdoorFit(
        tuple(float(v) for v in beta),
        float(np.clip(phi, 0.5, 0.98)),
        float(np.std(residual, ddof=1)),
        on_reanalysis,
        weather_reference=weather_reference,
    )


def outdoor_paths(
    model: OutdoorFit,
    weather: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    origin: datetime,
    timezone: str,
    n_paths: int,
    rng: np.random.Generator,
    observed_c: float | None,
) -> NDArray[np.float32]:
    """Generate anchored hourly AR1 errors and interpolate each path to minutes.

    Args:
        model: Fitted MOS/residual model.
        weather: The available hourly forecast, covering the requested minutes.
        index: Minute grid beginning at the prediction origin's minute boundary.
        origin: Actual prediction origin.
        timezone: Local-hour zone for MOS.
        n_paths: Full ensemble size.
        rng: Deterministic generator.
        observed_c: Fresh selected outdoor, or None for stationary initial error.

    Returns:
        Finite full-ensemble minute temperatures. Without an observation the
        initial error is sampled, not silently set to zero.

    Raises:
        ValueError: If the available forecast does not cover the horizon.
    """
    mapped = model.mean(weather, timezone)
    if (
        not np.isfinite(mapped).all()
        or weather.index[0] > index[0]
        or weather.index[-1] < index[-1]
    ):
        raise HvacError("weather forecast does not cover outdoor path horizon")
    # Hourly error knots start at the actual origin; the last is beyond the path end.
    horizon = (index[-1].to_pydatetime() - origin).total_seconds() / 3600
    knots = np.arange(max(int(np.ceil(horizon)) + 1, 1) + 1, dtype=float)
    actual = (
        pd.DatetimeIndex(weather.index) - pd.Timestamp(origin)
    ).total_seconds().to_numpy() / 3600
    error = np.empty((n_paths, len(knots)))
    error[:, 0] = (
        observed_c - float(np.interp(0.0, actual, mapped))
        if observed_c is not None
        else rng.normal(0, model.stationary_sd_c, n_paths)
    )
    innovations = (
        rng.normal(size=(n_paths, len(knots) - 1))
        * model.stationary_sd_c
        * np.sqrt(1 - model.phi**2)
    )
    for h in range(1, len(knots)):
        error[:, h] = model.phi * error[:, h - 1] + innovations[:, h - 1]
    minute_offsets = (index - pd.Timestamp(origin)).total_seconds().to_numpy() / 3600
    whole = np.floor(np.maximum(minute_offsets, 0)).astype(int)
    fraction = np.maximum(minute_offsets, 0) - whole
    mean_min = np.interp(minute_offsets, actual, mapped)
    values = (
        mean_min[None, :]
        + error[:, whole] * (1 - fraction)
        + error[:, whole + 1] * fraction
    )
    return np.asarray(values, dtype=np.float32)
