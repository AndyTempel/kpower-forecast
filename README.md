# ☀️ KPower Forecast 📈

[![PyPI version](https://img.shields.io/pypi/v/kpower-forecast.svg)](https://pypi.org/project/kpower-forecast/)
[![Python versions](https://img.shields.io/pypi/pyversions/kpower-forecast.svg)](https://pypi.org/project/kpower-forecast/)
[![CI](https://github.com/akorenc/kpower-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/AndyTempel/kpower-forecast/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-grade solar production and power consumption forecasting.**

Built with [Facebook Prophet](https://facebook.github.io/prophet/) and powered by [Open-Meteo](https://open-meteo.com/). KPower Forecast provides a high-level API for training and predicting energy metrics with physics-informed corrections.

---

## ✨ Key Features

- 🔋 **Dual Mode**: Specialized logic for both **Solar Production** and **Energy Consumption**.
- 🌓 **Night Masking**: Physics-informed clamping using solar elevation to eliminate "ghost production" at night.
- 🌡️ **Weather Integration**: Automatic fetching and resampling of temperature, cloud cover, and radiation.
- 🌦️ **Adaptive Weather Correction**: Learns location/model-specific weather bias as forecast history accumulates.
- ⚡ **Optional Curtailment Limits**: Clips delivered solar forecasts to inverter or export limits when configured.
- 🤖 **Prophet Optimized**: Pre-configured regressors for maximum accuracy.
- 💾 **Smart Persistence**: Automatic serialization of models to skip retraining when possible.
- ❄️ **Heat Pump Mode**: Optional temperature correlation for energy consumption models.

---

## 🚀 Quick Start

### Installation

```bash
# Core package
pip install kpower-forecast

# With CLI support (recommended for interactive use)
pip install "kpower-forecast[cli]"

# With the local Nixtla hybrid ML forecasting backend
pip install "kpower-forecast[ml]"

# With NeuralForecast / AI forecasting support
pip install "kpower-forecast[ai]"
 ```

### 🖥️ CLI Usage

KPower Forecast comes with a powerful CLI for interactive forecasting and visualization.

```bash
# Forecast solar production using Home Assistant CSV export
# Supports different data categories: instant_energy, cumulative_energy, power
# Supports different units: kWh, Wh, kW, W
# Optional delivered-energy curtailment limits can be supplied in kW
kpower-forecast solar rooftop-1 46.05 14.50 -i history.csv --category power --unit W --inverter-limit 10 --export-limit 7 --horizon 7

# Forecast power consumption
kpower-forecast consumption main-meter 46.05 14.50 -i history.csv --category cumulative_energy --unit kWh --horizon 3 --heatpump
```

**CLI Features:**
- **Automatic HA Parsing**: Heuristic detection of `last_changed` and `state` columns.
- **Smart Data Normalization**: Handles meter readings (cumulative), power (kW/W), and instant energy.
- **Heat Pump Mode**: Enable `--heatpump` to correlate consumption with outdoor temperature.
- **Inconsistent Intervals**: Robustly handles measurements with non-uniform time gaps.
- **Rich Tables**: Beautiful daily summary tables in your terminal.
- **Terminal Graphs**: Instant visualization of forecasts and confidence intervals via `plotext`.

### ☀️ Solar Production Forecast (API)

```python
from kpower_forecast import KPowerForecast
from kpower_forecast.core import DataCategory, MeasurementUnit
import pandas as pd

# 1. Initialize for your location with specific data types
kp = KPowerForecast(
    model_id="rooftop_solar",
    latitude=46.0569,
    longitude=14.5058,
    forecast_type="solar",
    data_category=DataCategory.POWER,
    unit=MeasurementUnit.W,
    inverter_ac_limit_kw=10.0,
    grid_export_limit_kw=7.0,
)

# 2. Train with your history
# history_df = pd.DataFrame({'ds': [...], 'y': [...]})
# kp.train(history_df)

# 3. Predict the next 7 days
forecast = kp.predict(days=7)
print(forecast[['ds', 'yhat']].head())
```

### 🏠 Energy Consumption Forecast

```python
kp_cons = KPowerForecast(
    model_id="house_meter",
    latitude=46.0569,
    longitude=14.5058,
    forecast_type="consumption",
    heat_pump_mode=True # Accounts for heating/cooling loads
)
```

### 🧠 Optional ML Forecasting Add-on

The core `KPowerForecast` API remains optimized for lightweight Prophet-based
forecasting. For local classical ML workflows, install the `ml` extra and use
`KPowerMLForecast` from the optional namespace:

```python
from kpower_forecast.ml import KPowerMLForecast, MLBackendType, MLForecastType

kp_ml = KPowerMLForecast(
    model_id="house_meter_ml",
    latitude=46.0569,
    longitude=14.5058,
    forecast_type=MLForecastType.CONSUMPTION,
    backend=MLBackendType.NIXTLA_HYBRID,
)

# history_df = pd.DataFrame({'ds': [...], 'y': [...]})
# kp_ml.train(history_df, force=True)
forecast = kp_ml.predict(days=3)
print(forecast[["ds", "yhat", "yhat_lower_90", "yhat_upper_90"]].head())
```

The ML add-on uses Nixtla-compatible backends behind a small project-owned
backend interface. `NIXTLA_HYBRID` combines a `statsforecast` structural baseline
with an `mlforecast`/LightGBM residual learner and is suitable for local CPU-only
controller deployments. `NEURALFORECAST` is also wired as a selectable Nixtla
backend for users who install `kpower-forecast[ai]` and provide NeuralForecast
model objects in `backend_params`. Future foundation-model adapters can plug into
the same backend contract without changing the public API.

For PV forecasts, `inverter_ac_limit_kw` and `grid_export_limit_kw` cap the
predicted interval energy to account for inverter clipping and static export
curtailment. `predict(dynamic_export_limits=...)` also accepts a dataframe with
`ds` plus `export_limit_kw`, `grid_export_limit_kw`, `curtailment_limit_kw`, or
`limit_kw` for time-varying export controls.

---

## Passive thermal response model

`kpower_forecast.thermal` is a separate, lightweight heating-regime API. It
identifies a stable first-order effective response from pairs of **real** indoor
observations at arbitrary spacing. Every pair carries coverage-gated HVAC
electric input; by default a transition needs an `hvac_coverage_ratio` of at least 0.90
(the EMS adapter reports its least-covered five-minute bucket). `train_with_weather` obtains historical outdoor temperature
from this package's weather client and integrates it across each actual
observation interval; indoor targets are never interpolated onto a regular
grid. The effective gain is a building response coefficient, not COP.

Training weather combines archive dates with the forecast endpoint's recent
`past_days` window (one day by default, plus today). Future date padding never
reaches the archive endpoint. Strict thermal callers preserve missing weather
and reject uncovered observation intervals. Recent weather uses the forecast
cache expiry rather than the long-lived archive cache; relative forecast
requests also refresh at UTC midnight.

```python
from kpower_forecast.thermal import KPowerThermalForecast, ThermalObservedTransition

model = KPowerThermalForecast(
    model_id="thermal_aggregate",
    authority_fingerprint="site-and-zone-configuration-epoch",
    storage_path="./data",
    latitude=46.0569,
    longitude=14.5058,
)
# observations: list[ThermalObservedTransition] from genuine indoor endpoints
# diagnostics = model.train_with_weather(observations)
# model.save()  # only when fitted; load() checks lineage and contract
# forecast = model.predict_with_weather(
#     origin=aligned_utc_origin,
#     initial_temperature_c=fresh_real_indoor_c,
#     hvac_electric_power_w=existing_heating_forecast_w,
#     hvac_drive_source="legacy_heating_forecast",
# )
```

**Fit.** Transitions are linked into chains; a gap of up to 30 minutes
between two covered transitions (typically one rejected for HVAC coverage) is
bridged by filling its *inputs* from the neighbouring intervals, while the
reading that resumes the chain stays the real target. Overlapping 6-24 hour
windows starting at real readings are simulated with the RC model, and the
loss covers every real reading in each window (output error). The time constant
is a grid search with a weak log-normal prior (median 60 h); gain and offset
are bounded least squares on the simulated response. The one-step transition
fit is also computed (`transition_fit`). When it is within the hard limits and
replays the holdout better at the skill horizons, it is published instead and
`fit_method` reads `transition`; on the reference site this was the case for one
closed-loop TRV zone.

**Quality.** Acceptance needs the accepted-history gates (72 h, 40
transitions, excitation) and a holdout MAE at +3/+6/+12 h at least 5 % below
persistence on two horizons. Sensors quantised to 0.1-0.2 °C make one-step
error indistinguishable from persistence, so one-step metrics are reported but
do not decide. A fit with a constrained coefficient, a boundary or flat time
constant, or less history (at least 24 h) is published as
`quality="low_identifiability"` with 1.5x wider bands, unless it is worse than
persistence on the holdout, in which case it is not fitted. Hard limits
(2-240 h, gain at most 20 °C/kW, |offset| at most 12 °C) still apply.

**Artifacts.** `load()` restores an artifact of the same model ID, lineage
fingerprint, contract and config. One written by another package version or
weather configuration loads with `needs_retrain=True`; callers should retrain
on the same history rather than discard the fit.

**Derived and naive forecasts.** `predict(..., equilibrium_shift_c=...)` and
`evaluate_with_weather(..., equilibrium_shift_c=...)` let a caller derive a
zone from the aggregate fit and score that against the zone's own fit.
`predict_naive` holds the current reading while fading a damped recent trend
(`recent_trend_c_per_hour`); it is the model-free last fallback.

Training diagnostics include rejected durations/coverage, excitation, fitted
time constant and gain, chronological holdout replay at 1/3/6/12 hours,
persistence comparison, and empirical temperature error bands. EMS must
additionally validate real future origins before granting any electrical
forecast authority. The API never chooses HVAC modes or an HVAC electrical
schedule.

---

### Offline identification matrix (calibration only)

With the `cli` extra installed, run:

```bash
python -m kpower_forecast.thermal.identification private-bundle.json private-report.json
```

`MatrixBundle` is a strict Pydantic boundary. Its frozen manifest declares the
model/source authority epoch, export start, fit cutoff, evaluation start (at least
six hours after fitting), final export cutoff, input provenance, target tolerance
and documented source confidence. Aware input timestamps normalize to UTC before
window arithmetic and serialization. `series` contains all keys `5`, `10`, `15`, `30`,
`60`, even if empty. Each series is an independently admitted minimum-separation
selection of genuine endpoints, with exact elapsed time, covered HP electrical
exposure and pre-exported outdoor means. `source_authority_id` must describe the
same selected authority at both endpoints. Data adapters own intermediate source,
receipt, quality and exposure checks; this tool cannot reconstruct them from rows.
Never rename five-minute rows to manufacture a longer-resolution experiment.

The report includes all 7/14/21-day configurations and an unweighted/runtime
baseline versus diagnostic confidence/elapsed weighting. Unknown sensor accuracy
uses a named manifest assumption (default 0.1 C standard deviation); documented
quantization contributes `resolution_c**2/12`. The conservative two-endpoint
variance bound does not assume independent endpoint errors. Confidence ratios
are capped before elapsed-duration weighting. Precision does not establish accuracy.

The solver and physical acceptance limits are shared with the existing RC model.
Rejected boundary/sign optima remain visible as diagnostics; tau remains bounded
at 240 h. Neighbor comparisons expose coefficient changes and sign agreement,
including rejected fits. Complete requested history must be available within the
authority/export boundary; short datasets are explicitly unavailable.

All physically accepted candidates replay the same chronological real calibration
targets at 30 min, 1 h, 2 h and 4 h, using observed input exposure and a persistence
baseline. A source/gap/value discontinuity breaks replay; no indoor interpolation
is allowed. Unmatched targets stay unmatched. These are observed-drive response
checks, **not operational unseen-origin forecast evidence**. Internal blocked
holdout status is reported separately; it cannot establish empirical reliability.

This first slice excludes calibrated excitation/cycle scoring, disturbance
classification, cycle-balanced weighting, final holdout selection and live-manager
integration. It never saves an operational model, requests weather, modifies a
controller or grants control/electrical authority. Keep private bundles/reports
outside Git. Identical CLI reruns are harmless; different existing outputs are
refused. Reports are completed and flushed in a private temporary sibling, then
atomically published without replacing an existing final report.

## Time and history contract

Forecast timestamps and the `ds` grid remain UTC. ML calendar and holiday features are derived
temporarily in the configured IANA timezone. Artifact contract version 3 records that timezone and
history policy version 2, so incompatible artifacts are retrained before prediction.

Callers can enable `preserve_gaps`. Power samples are then converted independently using the fixed
interval duration, cumulative energy is differenced only across adjacent valid samples, and missing
target intervals remain missing through normalization and feature construction.

Prediction weather for an elapsed prefix combines archive and recent-forecast weather. Only the
seam between the two sources, at most one hour (the hourly archive ends at 23:00 while the recent
forecast starts at 00:00), is interpolated in time; any other missing weather slot still fails
grid alignment.

Telemetry with intermittent dropouts can set `max_bridged_gap_intervals` (default `0`, off;
requires `preserve_gaps=True`, power or instant-energy input and a non-solar target). Interior
missing runs up to that many intervals are bridged linearly after normalization so a single dropped
row no longer splits the latest structural segment. Longer, leading and trailing gaps stay missing.
Bridged rows are excluded from weather-bias fitting and conformal calibration, the persisted
baseline history keeps the original gaps, and the manifest records the limit and bridged row count.

### Hybrid structure (`hybrid_structure`)

The Nixtla hybrid backend has two structures for non-solar targets:

- `recursive_seasonal_naive` (default): a seasonal-naive structural forecast (the last observed
  day, repeated) corrected by a LightGBM residual model fed its own lags recursively. It suits
  smooth series. For loads with irregular events, such as an on/off heat pump, it replays one
  day's events into every future day, and the recursion can oscillate over multi-day horizons.
- `profile_direct`: a mean profile per local quarter-hour and weekday class over the last
  `profile_lookback_days` (default 28), plus a regularised LightGBM residual model that predicts
  every future row directly from weather and calendar features. It uses no lags and no
  recursion, so a 5-day forecast does not drift or oscillate with lead time. Not available for
  solar.

Each weekday-class profile is shrunk toward the all-days profile with weight
`n / (n + profile_class_prior_days)` (default 4), where `n` is the number of days of that class in
the lookback. A weekend seen only twice then leans on the all-days profile instead of replaying
those two days' heat-pump runs, and a class with ample history keeps its own shape (weekday and
weekend household load differ). `profile_smoothing_minutes` (default 0, a multiple of the
interval) applies a centred ±minutes circular moving average over local time of day.

On a site with a fixed-speed heat pump (12 rolling origins, RMSE at +24/+120 h), `profile_direct`
improved whole-site consumption from 1100/1404 W to 976/1027 W and heating from 1046/1170 W to
832/907 W. The structure and lookback are part of artifact compatibility: a stored artifact
trained with other values (including the shrinkage and smoothing settings) is not restored on construction, `train(force=True)` retrains it, and
a non-forced `train()` raises "requires a full retrain", like the other compatibility settings.

### Rolling-origin candidate selection

Consumption and HVAC targets can set `candidate_selection=True` (default off; not for solar).
Training then backtests the candidates in `selection_candidates` (default all four) on day-ahead
windows: from local midnight of each of the last `selection_backtest_days` days (default 7), each
candidate forecasts the next `selection_horizon_hours` (default 24) after training only on rows
before that midnight, with the same historical weather for all. This is how load forecasts are
used and is far less noisy than a single holdout split. The candidates:

- `kpower_ml`: the configured ML backend, refitted per origin (always scored; ties keep it);
- `degree_hour_regression`: ridge regression with per-local-hour intercepts,
  `max(0, regression_base_temperature_c − T_out)` (default 16 °C) and
  `regression_extra_features` (default shortwave radiation);
- `local_slot_weekday_class_median`: the leakage-safe slot/weekday median;
- `ml_regression_blend`: `w·ML + (1 − w)·regression`, with `w` in [0, 1] fitted by least squares
  on the backtest (each origin is scored with a weight fitted on the other origins).

The candidates are ranked by RMSE of 1 h means taken at every step within each window. A run
forecast 15 minutes early costs a quarter of a run instead of a miss and a phantom run, and no
bin edge splits a near miss. A candidate whose mean error exceeds `selection_bias_tolerance`
(default 15 %) of the mean actual load, or `selection_bias_floor_kw` (default 0.05 kW) if that
is larger, cannot win. RMSE alone can prefer a smooth forecast that misses a third of the energy,
which an energy planner integrates. If every candidate is outside the guard, the least biased one
wins and `selection_reason` is `no_candidate_within_bias_guard`. With fewer than
`selection_min_origins` usable origins (default 3), the ML model is kept and
`selection_reason` is `backtest_too_short`.

Prediction intervals are calibrated on leave-one-origin-out residuals: each origin's residual
comes from the candidate the rule picks from the other origins. That calibrates the procedure
that is served, on residuals that did not choose it. `selected_candidate`, `candidate_metrics`
(`rmse_1h`, `rmse`, `mae`, `bias`, `mean_actual`, `rows`, `origins`), the blend weight and the
regression coefficients are persisted in the manifest. An artifact trained with different
selection settings is not restored and needs `train(force=True)`.

Backtesting refits the ML backend once per origin, so training costs about
`selection_backtest_days` extra ML fits. As history grows, the ML model improves and wins the
backtest without configuration changes.

### Known covariates

`known_covariates` names numeric columns that the caller knows in advance, such as a scheduled
HVAC mode or a thermostat target. Training history must contain them, and their names must not
repeat a weather column or a generated feature (`hour_sin`, `heating_degree`, …). They are
averaged onto the model grid and join the weather and calendar features. Training rows without a
value count as 0, like other missing features, so supply complete history.
`predict(known_future=...)` and `get_prediction_intervals(known_future=...)` must then supply
`ds` and every covariate for each model-grid row: from the first slot after training, not only
from `origin`, through the returned horizon. A missing or non-finite value raises
`ForecastAlignmentError`, and nothing is zero-filled. To let the degree-hour regression use a
covariate, add it to `regression_extra_features`. Changing `known_covariates` requires a full
retrain.

---

## 🛠️ Advanced Configuration

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | `str` | *required* | Unique ID for model persistence |
| `latitude` | `float` | *required* | Location Latitude |
| `longitude` | `float` | *required* | Location Longitude |
| `interval_minutes`| `int" | `15` | Data resolution (15 or 60) |
| `storage_path` | `str` | `"./data"` | Directory for saved models |
| `heat_pump_mode` | `bool` | `False` | Enable temperature regressor for consumption |
| `adaptive_weather_correction` | `bool` | `True` | Learn weather correction from archived forecasts, with historical weather fallback |
| `inverter_ac_limit_kw` | `float \| None` | `None` | Optional inverter AC output limit in kW |
| `grid_export_limit_kw` | `float \| None` | `None` | Optional grid export limit in kW |

Adaptive weather correction is conservative on new sites. Initial training works without historical forecast snapshots by falling back to archive weather, then improves as generated forecasts are archived and later matched with actual production.

---

## 🔢 Versioning

This project follows a custom **Date-Based Versioning** scheme:
`YYYY.MM.Patch` (e.g., `2026.2.1`)

- **YYYY**: Year of release.
- **MM**: Month of release (no leading zero, 1-12).
- **Patch**: Incremental counter for releases within the same month.

### Enforcement
- **CI Validation**: Every Pull Request is checked against `scripts/validate_version.py` to ensure adherence.
- **Consistency**: Both `pyproject.toml` and `src/kpower_forecast/__init__.py` must match exactly.

---

## 🧪 Development & Testing

We use [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

```bash
# Clone and setup
git clone https://github.com/akorenc/kpower-forecast
cd kpower-forecast
uv sync --all-extras

# Run tests
uv run pytest

# Linting
uv run ruff check .
```

---

## 📄 License

Distributed under the **GNU Affero General Public License v3.0**. See `LICENSE` for more information.

---
<p align="center">Made with ❤️ for a greener future.</p>
