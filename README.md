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
