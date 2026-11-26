"""Snow on the PV panels: a load model stepped through the weather.

Fresh snow covers a PV array completely, so the array produces nothing until
the snow sheds, and the panels stay covered for longer than the snow on the
ground suggests. Ground snow depth is therefore not used. The model keeps one
state, the snow load on the panel in cm:

* snowfall adds to the load while the air is at or below
  ``accumulation_max_temperature_c``;
* warm air (per degree-hour above 0 °C) and light on the dark panel (per
  kWh/m² of global horizontal irradiance) shed it;
* the output factor falls linearly from 1 at no load to 0 at ``full_cm``.

Measured output overrides the weather where it disagrees (rime or drifting that
the weather cannot show, or snow that shed early): near-zero output under
enough light marks the panels covered, normal output marks them clear.

Snowfall in the weather frames is a per-hour rate (see
``kpower_forecast.weather_client.WEATHER_ACCUMULATION_UNITS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# The panel is fully clear only below this load; float noise is not cover.
_CLEAR_LOAD_CM: float = 1e-6


class SnowCoverStatus(str, Enum):
    """Whether snow lies on the panels."""

    CLEAR = "clear"
    PARTIAL = "partial"
    COVERED = "covered"


class SnowCoverSource(str, Enum):
    """What last set the snow load."""

    WEATHER = "weather"
    OBSERVED = "observed"


class SnowCoverParameters(BaseModel):
    """Snow-load model parameters.

    The defaults were fitted on one winter of measured daily output from a
    low-tilt residential array, against archived forecasts and reanalysis
    alike.
    """

    accumulation_max_temperature_c: float = Field(default=1.5, ge=-10.0, le=5.0)
    melt_cm_per_degree_hour: float = Field(default=0.1, ge=0.0, le=5.0)
    shed_cm_per_kwh_m2: float = Field(default=1.0, ge=0.0, le=50.0)
    full_cm: float = Field(default=1.0, gt=0.0, le=50.0)
    # Observed cover: light enough to expect at least this mean power (kW).
    observed_min_expected_kw: float = Field(default=0.3, gt=0.0)
    # Covered: measured output at most this share of the expected output...
    observed_covered_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    # ...for this long, at or below this air temperature.
    observed_covered_minutes: int = Field(default=60, ge=15, le=360)
    observed_max_temperature_c: float = Field(default=3.0, ge=-10.0, le=10.0)
    # Clear: measured output at least this share of the expected output.
    observed_clear_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    observed_clear_minutes: int = Field(default=30, ge=15, le=360)


@dataclass(frozen=True)
class SnowLoadState:
    """Snow load on the panels at the end of a weather row."""

    load_cm: float
    source: SnowCoverSource
    at: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation.

        Returns:
            ``load_cm``, ``source`` and ``at`` (ISO 8601), as read by
            :meth:`from_dict`.
        """
        return {
            "load_cm": float(self.load_cm),
            "source": self.source.value,
            "at": self.at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SnowLoadState:
        """Restore a state written by :meth:`to_dict`.

        Raises:
            ValueError: If a field is missing or invalid.
        """
        try:
            load = float(payload["load_cm"])
            source = SnowCoverSource(payload["source"])
            at = pd.Timestamp(payload["at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid snow load state: {exc}") from exc
        if not np.isfinite(load) or load < 0.0 or at.tzinfo is None:
            raise ValueError("invalid snow load state")
        return cls(load_cm=load, source=source, at=at.tz_convert("UTC"))


@dataclass(frozen=True)
class SnowLoadTrajectory:
    """Snow load and output factor per weather row."""

    ds: pd.Series
    load_cm: np.ndarray
    factor: np.ndarray
    source: list[SnowCoverSource]

    def final_state(self) -> SnowLoadState:
        """State after the last row.

        Raises:
            ValueError: If the trajectory is empty.
        """
        if len(self.load_cm) == 0:
            raise ValueError("empty snow load trajectory")
        return SnowLoadState(
            load_cm=float(self.load_cm[-1]),
            source=self.source[-1],
            at=pd.Timestamp(self.ds.iloc[-1]),
        )


def cover_factor(load_cm: np.ndarray, full_cm: float) -> np.ndarray:
    """Output factor for a snow load: 1 when clear, 0 at ``full_cm`` or more.

    Args:
        load_cm: Snow load per row, in cm.
        full_cm: Load at which the panels give no output.

    Returns:
        Factor per row in ``[0, 1]``, falling linearly between clear and full.
    """
    load = np.asarray(load_cm, dtype=float)
    factor = np.clip(1.0 - load / full_cm, 0.0, 1.0)
    return np.where(load <= _CLEAR_LOAD_CM, 1.0, factor)


def cover_status(load_cm: float, full_cm: float) -> SnowCoverStatus:
    """Classify a snow load.

    Args:
        load_cm: Snow load in cm.
        full_cm: Load at which the panels give no output.

    Returns:
        Clear, partial or covered.
    """
    if load_cm <= _CLEAR_LOAD_CM:
        return SnowCoverStatus.CLEAR
    if load_cm >= full_cm:
        return SnowCoverStatus.COVERED
    return SnowCoverStatus.PARTIAL


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame))
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.asarray(np.nan_to_num(values, nan=0.0), dtype=float)


def _run_mask(condition: np.ndarray, min_rows: int) -> np.ndarray:
    """Rows inside a run of ``condition`` at least ``min_rows`` long, from the
    row on which the run reaches that length (a run is known only then)."""
    mask = np.zeros(len(condition), dtype=bool)
    run = 0
    for index, value in enumerate(condition):
        run = run + 1 if value else 0
        if run >= min_rows:
            mask[index] = True
    return mask


def observed_cover(
    actual_kwh: pd.Series,
    expected_kwh: np.ndarray,
    temperature_c: np.ndarray,
    interval_minutes: int,
    parameters: SnowCoverParameters,
) -> np.ndarray:
    """Per-row observation: +1 covered, -1 clear, 0 no evidence.

    Args:
        actual_kwh: Measured interval energy; missing rows give no evidence.
        expected_kwh: Interval energy expected without snow.
        temperature_c: Air temperature per row.
        interval_minutes: Row length.
        parameters: Snow-cover parameters.

    Returns:
        Integer array aligned with ``actual_kwh``.
    """
    actual = pd.to_numeric(actual_kwh, errors="coerce").to_numpy(dtype=float)
    measured = np.isfinite(actual)
    expected = np.asarray(expected_kwh, dtype=float)
    lit = expected >= parameters.observed_min_expected_kw * interval_minutes / 60.0
    covered_rows = (
        measured
        & lit
        & (actual <= parameters.observed_covered_ratio * expected)
        & (
            np.asarray(temperature_c, dtype=float)
            <= parameters.observed_max_temperature_c
        )
    )
    clear_rows = measured & lit & (actual >= parameters.observed_clear_ratio * expected)
    covered = _run_mask(
        covered_rows, ceil(parameters.observed_covered_minutes / interval_minutes)
    )
    clear = _run_mask(
        clear_rows, ceil(parameters.observed_clear_minutes / interval_minutes)
    )
    observation = np.zeros(len(actual), dtype=int)
    observation[covered] = 1
    observation[clear] = -1
    return observation


def step_snow_load(
    weather: pd.DataFrame,
    interval_minutes: int,
    parameters: SnowCoverParameters,
    initial: SnowLoadState | None = None,
    observation: np.ndarray | None = None,
) -> SnowLoadTrajectory:
    """Step the panel snow load through weather rows.

    Args:
        weather: Rows with ``ds`` and, where available, ``snowfall`` (cm/h),
            ``temperature_2m`` (°C) and ``shortwave_radiation`` (W/m²).
            Missing columns count as zero.
        interval_minutes: Row length.
        parameters: Snow-cover parameters.
        initial: Load before the first row; clear when omitted.
        observation: Optional :func:`observed_cover` result for the rows. An
            observed-covered row raises the load to at least ``full_cm``, an
            observed-clear row resets it to zero.

    Returns:
        Load, factor and source after each row.

    Raises:
        ValueError: If ``observation`` does not match the rows.
    """
    rows = len(weather)
    if observation is not None and len(observation) != rows:
        raise ValueError("observation must align with the weather rows")
    hours = interval_minutes / 60.0
    snowfall = np.clip(_numeric(weather, "snowfall"), 0.0, None)
    temperature = _numeric(weather, "temperature_2m")
    ghi = np.clip(_numeric(weather, "shortwave_radiation"), 0.0, None)
    accumulation = np.where(
        temperature <= parameters.accumulation_max_temperature_c, snowfall * hours, 0.0
    )
    shed = (
        parameters.melt_cm_per_degree_hour * np.clip(temperature, 0.0, None)
        + parameters.shed_cm_per_kwh_m2 * ghi / 1000.0
    ) * hours

    load = 0.0 if initial is None else initial.load_cm
    source = SnowCoverSource.WEATHER if initial is None else initial.source
    loads = np.zeros(rows)
    sources: list[SnowCoverSource] = []
    for index in range(rows):
        if accumulation[index] > 0.0:
            source = SnowCoverSource.WEATHER
        load = max(0.0, load + accumulation[index] - shed[index])
        if observation is not None and observation[index] > 0:
            load = max(load, parameters.full_cm)
            source = SnowCoverSource.OBSERVED
        elif observation is not None and observation[index] < 0:
            load = 0.0
            source = SnowCoverSource.OBSERVED
        loads[index] = load
        sources.append(source)
    return SnowLoadTrajectory(
        ds=pd.to_datetime(weather["ds"], utc=True).reset_index(drop=True),
        load_cm=loads,
        factor=cover_factor(loads, parameters.full_cm),
        source=sources,
    )


def pessimistic_parameters(parameters: SnowCoverParameters) -> SnowCoverParameters:
    """Parameters for the lower band edge: more snow that sheds more slowly.

    Args:
        parameters: Parameters for the point forecast.

    Returns:
        A copy that accumulates up to 1 °C warmer, melts and sheds at half the
        rate and counts as covered at half the load.
    """
    return parameters.model_copy(
        update={
            "accumulation_max_temperature_c": parameters.accumulation_max_temperature_c
            + 1.0,
            "melt_cm_per_degree_hour": parameters.melt_cm_per_degree_hour / 2.0,
            "shed_cm_per_kwh_m2": parameters.shed_cm_per_kwh_m2 / 2.0,
            "full_cm": parameters.full_cm / 2.0,
        }
    )


def energy_per_irradiance(
    actual_kwh: pd.Series, ghi_w_m2: np.ndarray, interval_minutes: int
) -> float | None:
    """Median measured kWh per kWh/m² of horizontal irradiance.

    A crude expected output for :func:`observed_cover`; a covered row's
    measured output is a few percent of any working array's, so the sun-angle
    dependence it ignores does not matter there.

    Returns:
        The median over rows with at least 100 W/m² and output, or ``None``
        without such rows.
    """
    actual = pd.to_numeric(actual_kwh, errors="coerce").to_numpy(dtype=float)
    ghi = np.asarray(ghi_w_m2, dtype=float)
    usable = np.isfinite(actual) & (ghi >= 100.0) & (actual > 0.0)
    if not usable.any():
        return None
    irradiance_kwh = ghi[usable] * interval_minutes / 60.0 / 1000.0
    return float(np.median(actual[usable] / irradiance_kwh))
