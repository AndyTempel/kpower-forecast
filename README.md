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

# With optional Nixtla-based ML forecasting backends
pip install "kpower-forecast[ml]"
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
forecasting. For heavier workstation/server training workflows, install the ML
extra and use `KPowerMLForecast` from the optional namespace:

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
with an `mlforecast`/LightGBM residual learner. `NEURALFORECAST` is also wired as
a selectable Nixtla backend for users who provide NeuralForecast model objects in
`backend_params`. Future foundation-model adapters can plug into the same backend
contract without changing the public API.

For PV forecasts, `inverter_ac_limit_kw` and `grid_export_limit_kw` cap the
predicted interval energy to account for inverter clipping and static export
curtailment. `predict(dynamic_export_limits=...)` also accepts a dataframe with
`ds` plus `export_limit_kw`, `grid_export_limit_kw`, `curtailment_limit_kw`, or
`limit_kw` for time-varying export controls.

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
