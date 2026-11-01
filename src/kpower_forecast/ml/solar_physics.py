"""Physical PV baseline for the ``irradiance_direct`` solar structure.

The baseline is the irradiance on the panel plane times a learned yield per
sun position. The plane (tilt, azimuth) and the alignment of the weather
radiation to the target interval are fitted from history, so no site
configuration is needed. Everything here is a pure function of the weather and
the fitted state; nothing depends on the previous day's output.

Radiation from Open-Meteo is a mean over the preceding period, archive data is
hourly and interpolated, and the target covers the following interval. The
fitted ``lag_steps`` absorbs that offset for whichever source trained the
model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

# Ground reflectance for the isotropic transposition (grass/soil).
ALBEDO = 0.2
# Beam irradiance is not resolved for a sun this close to the horizon.
_MAX_ZENITH_DEG = 87.0
_MAX_DNI_W_M2 = 1100.0
# Rows used to fit the plane and the yield map: sun above this elevation and
# a sky at least this clear (shortwave over clear-sky GHI).
_FIT_MIN_ELEVATION_DEG = 8.0
_FIT_MIN_CLEAR_SKY_INDEX = 0.6
_MIN_FIT_ROWS = 96
_TILTS = np.arange(0.0, 61.0, 10.0)
_AZIMUTHS = np.arange(90.0, 271.0, 20.0)
_LAGS = (-2, -1, 0, 1, 2)
# Yield map bins over sun position; sparse bins shrink toward the global
# yield with weight n / (n + prior).
AZIMUTH_BIN_DEG = 15.0
ELEVATION_BIN_DEG = 5.0
_YIELD_PRIOR_ROWS = 20.0


@dataclass(frozen=True, slots=True)
class PanelGeometry:
    """Fitted panel plane and radiation alignment.

    Attributes:
        tilt_deg: Plane tilt from horizontal.
        azimuth_deg: Plane azimuth, degrees clockwise from north (180 = south).
        lag_steps: Rows the radiation is shifted against the target; positive
            uses later radiation rows.
        fitted: Whether the values came from history (else the horizontal
            fallback).
    """

    tilt_deg: float = 0.0
    azimuth_deg: float = 180.0
    lag_steps: int = 0
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PanelGeometry:
        """Build a geometry from :meth:`to_dict` output."""
        return cls(
            tilt_deg=float(payload.get("tilt_deg", 0.0)),
            azimuth_deg=float(payload.get("azimuth_deg", 180.0)),
            lag_steps=int(payload.get("lag_steps", 0)),
            fitted=bool(payload.get("fitted", False)),
        )


def sun_position(
    ds: pd.Series, latitude: float, longitude: float, interval_minutes: int
) -> pd.DataFrame:
    """Sun zenith, azimuth and elevation at the middle of each target interval.

    Args:
        ds: Interval start timestamps.
        latitude: Site latitude.
        longitude: Site longitude.
        interval_minutes: Target interval length.

    Returns:
        Frame with ``zenith``, ``azimuth`` and ``elevation`` in degrees and the
        ``day_of_year``, indexed like ``ds``.
    """
    from pvlib.solarposition import get_solarposition

    middle = pd.DatetimeIndex(pd.to_datetime(ds, utc=True)) + pd.Timedelta(
        minutes=interval_minutes / 2
    )
    position = get_solarposition(middle, latitude, longitude)
    return pd.DataFrame(
        {
            "zenith": position["apparent_zenith"].to_numpy(dtype=float),
            "azimuth": position["azimuth"].to_numpy(dtype=float),
            "elevation": position["apparent_elevation"].to_numpy(dtype=float),
            "day_of_year": np.asarray(middle.dayofyear, dtype=float),
        },
        index=ds.index,
    )


def _column(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame[name], errors="coerce")
    return None if values.isna().all() else values


def irradiance_components(
    weather: pd.DataFrame, sun: pd.DataFrame, lag_steps: int = 0
) -> pd.DataFrame:
    """Global, beam-normal and diffuse irradiance per row.

    Uses the weather's direct/diffuse split when present, else the Erbs
    decomposition of the global horizontal irradiance.

    Args:
        weather: Rows with ``shortwave_radiation`` and optionally
            ``direct_radiation`` and ``diffuse_radiation`` (W/m²).
        sun: Output of :func:`sun_position` for the same rows.
        lag_steps: Shift of the radiation rows (see :class:`PanelGeometry`).

    Returns:
        Frame with ``ghi``, ``dni`` and ``dhi`` (W/m²), indexed like
        ``weather``.

    Raises:
        ValueError: If ``shortwave_radiation`` is missing.
    """
    ghi = _column(weather, "shortwave_radiation")
    if ghi is None:
        raise ValueError("irradiance needs shortwave_radiation")
    direct = _column(weather, "direct_radiation")
    diffuse = _column(weather, "diffuse_radiation")
    frame = pd.DataFrame({"ghi": ghi}, index=weather.index)
    if direct is not None and diffuse is not None:
        frame["direct"] = direct
        frame["diffuse"] = diffuse
    if lag_steps:
        frame = frame.shift(-lag_steps).ffill().bfill()
    frame = frame.fillna(0.0).clip(lower=0.0)
    zenith = sun["zenith"].to_numpy(dtype=float)
    cos_zenith = np.cos(np.radians(zenith))
    resolved = zenith < _MAX_ZENITH_DEG
    if "direct" in frame.columns:
        dni = np.where(
            resolved, frame["direct"].to_numpy() / np.maximum(cos_zenith, 1e-3), 0.0
        )
        dhi = frame["diffuse"].to_numpy(dtype=float)
    else:
        from pvlib.irradiance import erbs

        split = erbs(
            frame["ghi"],
            pd.Series(zenith, index=frame.index),
            sun["day_of_year"].to_numpy(dtype=float),
        )
        dni = np.where(resolved, np.asarray(split["dni"], dtype=float), 0.0)
        dhi = np.asarray(split["dhi"], dtype=float)
    return pd.DataFrame(
        {
            "ghi": frame["ghi"].to_numpy(dtype=float),
            "dni": np.clip(np.nan_to_num(dni), 0.0, _MAX_DNI_W_M2),
            "dhi": np.clip(np.nan_to_num(dhi), 0.0, None),
        },
        index=weather.index,
    )


def plane_of_array(
    components: pd.DataFrame, sun: pd.DataFrame, tilt_deg: float, azimuth_deg: float
) -> np.ndarray:
    """Isotropic-sky irradiance on a tilted plane (W/m²).

    Args:
        components: Output of :func:`irradiance_components`.
        sun: Output of :func:`sun_position` for the same rows.
        tilt_deg: Plane tilt from horizontal.
        azimuth_deg: Plane azimuth, degrees clockwise from north.

    Returns:
        Plane-of-array global irradiance per row.
    """
    tilt = np.radians(tilt_deg)
    zenith = np.radians(sun["zenith"].to_numpy(dtype=float))
    azimuth = np.radians(sun["azimuth"].to_numpy(dtype=float))
    cos_aoi = np.cos(zenith) * np.cos(tilt) + np.sin(zenith) * np.sin(tilt) * np.cos(
        azimuth - np.radians(azimuth_deg)
    )
    beam = components["dni"].to_numpy() * np.clip(cos_aoi, 0.0, None)
    sky = components["dhi"].to_numpy() * (1.0 + np.cos(tilt)) / 2.0
    ground = components["ghi"].to_numpy() * ALBEDO * (1.0 - np.cos(tilt)) / 2.0
    return np.asarray(np.clip(beam + sky + ground, 0.0, None), dtype=float)


def clear_rows(
    weather: pd.DataFrame, sun: pd.DataFrame, clear_sky_ghi: pd.Series
) -> np.ndarray:
    """Rows clear and high enough to fit the plane and the yield map."""
    ghi = pd.to_numeric(weather["shortwave_radiation"], errors="coerce").to_numpy()
    clear = np.asarray(clear_sky_ghi, dtype=float)
    index = np.divide(ghi, clear, out=np.zeros_like(ghi), where=clear > 1.0)
    return np.asarray(
        (sun["elevation"].to_numpy() > _FIT_MIN_ELEVATION_DEG)
        & (index > _FIT_MIN_CLEAR_SKY_INDEX),
        dtype=bool,
    )


def fit_geometry(
    target: pd.Series,
    weather: pd.DataFrame,
    sun: pd.DataFrame,
    mask: np.ndarray,
) -> PanelGeometry:
    """Search the plane and radiation lag that best explain the target.

    The score is the R² of ``target ≈ k · POA`` through the origin on the
    masked rows; a coarse grid over plane and lag is refined by a pattern
    search over the plane.

    Args:
        target: Measured interval energy, aligned with ``weather``.
        weather: Weather rows (see :func:`irradiance_components`).
        sun: Sun position for the same rows.
        mask: Rows used for fitting (see :func:`clear_rows`).

    Returns:
        The fitted geometry, or the horizontal fallback with too few rows.
    """
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    usable = mask & np.isfinite(y) & (y >= 0.0)
    if usable.sum() < _MIN_FIT_ROWS:
        return PanelGeometry()
    y_fit = y[usable]
    best: tuple[float, float, float, int] = (-np.inf, 0.0, 180.0, 0)
    components_by_lag = {
        lag: irradiance_components(weather, sun, lag).loc[usable] for lag in _LAGS
    }
    sun_fit = sun.loc[usable]

    def score(tilt: float, azimuth: float, lag: int) -> float:
        poa = plane_of_array(components_by_lag[lag], sun_fit, tilt, azimuth)
        denominator = float(poa @ poa)
        if denominator <= 0.0:
            return -np.inf
        k = float(poa @ y_fit) / denominator
        residual = y_fit - k * poa
        total = float(((y_fit - y_fit.mean()) ** 2).sum())
        return 1.0 - float(residual @ residual) / total if total > 0 else -np.inf

    for lag in _LAGS:
        for tilt in _TILTS:
            for azimuth in _AZIMUTHS:
                value = score(float(tilt), float(azimuth), lag)
                if value > best[0]:
                    best = (value, float(tilt), float(azimuth), lag)
    # Pattern search from the coarse optimum: tilt and azimuth trade off, so
    # one fixed refinement window can miss the true plane.
    value, tilt, azimuth, lag = best
    tilt_step, azimuth_step = 10.0, 20.0
    while tilt_step >= 0.5:
        moved = False
        for d_tilt in (-tilt_step, 0.0, tilt_step):
            for d_azimuth in (-azimuth_step, 0.0, azimuth_step):
                candidate_tilt = float(np.clip(tilt + d_tilt, 0.0, 90.0))
                candidate = score(candidate_tilt, azimuth + d_azimuth, lag)
                if candidate > value + 1e-12:
                    value, tilt, azimuth = (
                        candidate,
                        candidate_tilt,
                        azimuth + d_azimuth,
                    )
                    moved = True
        if not moved:
            tilt_step, azimuth_step = tilt_step / 2.0, azimuth_step / 2.0
    return PanelGeometry(
        tilt_deg=tilt, azimuth_deg=azimuth % 360.0, lag_steps=lag, fitted=True
    )


def sun_bin_keys(sun: pd.DataFrame) -> list[str]:
    """Yield-map key (azimuth bin, elevation bin) per row."""
    azimuth = np.floor(sun["azimuth"].to_numpy() / AZIMUTH_BIN_DEG).astype(int)
    elevation = np.floor(
        np.clip(sun["elevation"].to_numpy(), 0.0, None) / ELEVATION_BIN_DEG
    ).astype(int)
    return [f"{a}:{e}" for a, e in zip(azimuth, elevation, strict=True)]


def fit_yield_map(
    target: pd.Series, poa: np.ndarray, sun: pd.DataFrame, mask: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Interval energy per unit of plane irradiance, by sun position.

    The map captures what the plane model misses at a given sun position:
    horizon and near shading, incidence losses, inverter clipping. Bins are
    ratios of sums over clear rows, shrunk toward the global ratio.

    Args:
        target: Measured interval energy, aligned with ``poa``.
        poa: Plane-of-array irradiance per row.
        sun: Sun position for the same rows.
        mask: Rows used for fitting.

    Returns:
        The global yield and the per-bin yields.
    """
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    usable = mask & np.isfinite(y) & (y >= 0.0) & (poa > 30.0)
    if usable.sum() < _MIN_FIT_ROWS:
        usable = np.isfinite(y) & (y >= 0.0) & (poa > 30.0)
    if not usable.any():
        return 0.0, {}
    global_yield = float(y[usable].sum() / poa[usable].sum())
    keys = np.array(sun_bin_keys(sun.loc[usable]))
    frame = pd.DataFrame({"key": keys, "y": y[usable], "poa": poa[usable]})
    yields: dict[str, float] = {}
    for key, rows in frame.groupby("key"):
        n = float(len(rows))
        ratio = float(rows["y"].sum() / rows["poa"].sum())
        weight = n / (n + _YIELD_PRIOR_ROWS)
        yields[str(key)] = weight * ratio + (1.0 - weight) * global_yield
    return global_yield, yields


def baseline_energy(
    poa: np.ndarray, sun: pd.DataFrame, global_yield: float, yields: dict[str, float]
) -> np.ndarray:
    """Physical baseline: plane irradiance times the yield for the sun bin."""
    factors = np.array([yields.get(key, global_yield) for key in sun_bin_keys(sun)])
    baseline = np.asarray(poa, dtype=float) * factors
    baseline[sun["elevation"].to_numpy() <= 0.0] = 0.0
    return np.asarray(np.clip(baseline, 0.0, None), dtype=float)


def diffuse_fraction(weather: pd.DataFrame) -> np.ndarray:
    """Diffuse share of horizontal irradiance; 1.0 without a direct/diffuse split."""
    direct = _column(weather, "direct_radiation")
    diffuse = _column(weather, "diffuse_radiation")
    if direct is None or diffuse is None:
        return np.ones(len(weather))
    total = (direct.fillna(0.0) + diffuse.fillna(0.0)).to_numpy(dtype=float)
    share = np.divide(
        diffuse.fillna(0.0).to_numpy(dtype=float),
        total,
        out=np.ones_like(total),
        where=total > 1.0,
    )
    return np.asarray(np.clip(share, 0.0, 1.0), dtype=float)
