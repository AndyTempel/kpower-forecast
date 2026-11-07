"""Bundle identity and availability boundaries, without numerical backends."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from kpower_forecast.hvac import HvacEvaluationBundle


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Create a tiny bundle with one observed zero, a gap and a future spike."""
    minutes = pd.DataFrame(
        {"power_w": [0.0, float("nan"), 9000.0]},
        index=pd.date_range("2026-11-07T00:00:00Z", periods=3, freq="min"),
    )
    minutes.to_parquet(tmp_path / "minutes.parquet")
    (tmp_path / "capabilities.json").write_text('{"family":"fixed_speed"}')
    manifest = {
        "exported_at": "2026-11-07T01:00:00Z",
        "start": "2026-11-07T00:00:00Z",
        "end": "2026-11-07T00:03:00Z",
        "controller_id": "test",
        "device_id": "hp",
        "timezone": "Europe/Ljubljana",
        "latitude": 46,
        "longitude": 14,
        "ems_commit": "1234567",
        "ingest_lag_min": 1,
        "files": {
            name: {
                "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
                "format": kind,
            }
            for name, kind in [
                ("minutes.parquet", "parquet"),
                ("capabilities.json", "json"),
            ]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_bundle_preserves_gaps_and_excludes_unavailable_minutes(
    bundle_dir: Path,
) -> None:
    """Future rows cannot affect an earlier-origin history, including at the cutoff."""
    bundle = HvacEvaluationBundle.load(bundle_dir)
    origin = datetime(2026, 11, 7, 0, 3, tzinfo=timezone.utc)
    history = bundle.available_at(origin)
    assert history.power_w.iloc[0] == 0
    assert pd.isna(history.power_w.iloc[1])
    assert len(history) == 2
    history.iloc[0, 0] = 123
    assert bundle.available_at(origin).power_w.iloc[0] == 0
    with pytest.raises(ValueError, match="UTC"):
        bundle.available_at(origin.replace(tzinfo=None))


@pytest.mark.parametrize("attack", ["hash", "escape", "unmeasured"])
def test_bundle_rejects_unverified_inputs(bundle_dir: Path, attack: str) -> None:
    """Unverified file identity, paths and availability cannot enter a model."""
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if attack == "hash":
        manifest["files"]["minutes.parquet"]["sha256"] = "0" * 64
    elif attack == "escape":
        manifest["files"]["../escape.json"] = manifest["files"]["capabilities.json"]
    else:
        manifest["ingest_lag_min"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle = HvacEvaluationBundle.load(bundle_dir)
        bundle.available_at(datetime(2026, 11, 7, 0, 3, tzinfo=timezone.utc))


def test_hvac_import_is_isolated_and_legacy_root_export_still_resolves() -> None:
    """A fresh interpreter must not pay for optional model backends to read inputs."""
    source = """import sys
import kpower_forecast.hvac
assert not any(name in sys.modules for name in (
 'prophet', 'cmdstanpy', 'pvlib', 'kpower_forecast.core', 'kpower_forecast.ml'))
from kpower_forecast import KPowerForecast
assert KPowerForecast.__name__ == 'KPowerForecast'
"""
    subprocess.run([sys.executable, "-c", source], check=True, env=os.environ.copy())
