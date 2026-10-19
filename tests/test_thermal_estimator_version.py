import hashlib
from pathlib import Path

import kpower_forecast
from kpower_forecast.thermal import THERMAL_ESTIMATOR_VERSION

# Thermal fitting, prediction and the weather they read. Callers keep authority
# evidence per THERMAL_ESTIMATOR_VERSION, so an unnoticed change here would
# inherit evidence the new estimator never earned.
_PACKAGE = Path(kpower_forecast.__file__).parent
_THERMAL = sorted((_PACKAGE / "thermal").glob("*.py"))
_SOURCES = [*_THERMAL, _PACKAGE / "weather_client.py"]


def _digest() -> str:
    digest = hashlib.sha256()
    for path in _SOURCES:
        digest.update(path.relative_to(_PACKAGE).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def test_thermal_sources_match_the_estimator_version() -> None:
    # On failure: if the change alters how temperature is fitted or predicted,
    # set THERMAL_ESTIMATOR_VERSION to the new release version; either way,
    # update the digest below.
    assert (THERMAL_ESTIMATOR_VERSION, _digest()) == (
        "2026.10.1",
        "93296a33444016b23c905e912cbb0ff5f684bc6adb3c97f6c09f65bbfd6e3e28",
    )
