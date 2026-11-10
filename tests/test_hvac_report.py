"""Gate decisions require identical eligible pairs and never turn gaps into passes."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from kpower_forecast.hvac._report import summarize

ORIGIN = datetime(2026, 11, 9, tzinfo=UTC)


def test_gates_pair_evidence_and_preserve_known_failure_among_unknown_buckets():
    runs = []
    for bucket, lead in (("0-15m", 0), ("15-30m", 15), ("30-60m", 30)):
        for model, probability in (("cycle", 0.2), ("B5", 0.3)):
            runs.append(
                {
                    "origin": ORIGIN,
                    "interval_start": ORIGIN + timedelta(minutes=lead),
                    "minutes": 5,
                    "lead_min": lead,
                    "bucket": bucket,
                    "model": model,
                    "eligible": True,
                    "probability": probability,
                    "actual_run": 0.0,
                    "brier": probability**2,
                }
            )
    # This bad cycle prediction has no eligible B5 mate, so it cannot change A1.
    runs.append(
        {
            **runs[0],
            "interval_start": ORIGIN + timedelta(minutes=5),
            "probability": 1.0,
            "brier": 1.0,
        }
    )
    quarters = pd.DataFrame(
        [
            {
                "origin": ORIGIN,
                "interval_start": ORIGIN + timedelta(hours=1),
                "bucket": "1-3h",
                "model": model,
                "eligible": True,
                "pinball": score,
                "crps": float("nan"),
            }
            for model, score in (("cycle", 1.0), ("B3", 0.9))
        ]
    )
    report = summarize({"runs": pd.DataFrame(runs), "quarters": quarters}, n_paths=256)
    assert report["gates"]["A1"]["status"] == "pass"
    assert [row["n"] for row in report["gates"]["A1"]["checks"]] == [1, 1, 1]
    assert report["gates"]["A2"]["status"] == "fail"
    assert report["gates"]["A4"]["status"] == "unscorable"
    assert report["gates"]["A5"]["status"] == "unscorable"
    assert report["gates"]["A7"]["status"] == "unscorable"
    duplicate = pd.concat(
        [pd.DataFrame(runs), pd.DataFrame(runs[:1])], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate score identity"):
        summarize({"runs": duplicate}, n_paths=256)


def test_official_energy_and_a4_exclude_better_later_windows():
    records = []
    for lead in (0, 24, 48):
        for model, error in (
            ("cycle", 10.0 if lead == 0 else 0.0),
            ("B1", 1.0 if lead == 0 else 20.0),
        ):
            records.append(
                {
                    "origin": ORIGIN,
                    "interval_start": ORIGIN + timedelta(hours=lead),
                    "duration_h": 24,
                    "model": model,
                    "eligible": True,
                    "actual_kwh": 100.0,
                    "error_kwh": error,
                    "partial_error_kwh": error,
                    "coverage": 1.0,
                }
            )
    report = summarize({"energy": pd.DataFrame(records)}, n_paths=256)
    cycle = next(row for row in report["M6"] if row["model"] == "cycle")
    assert cycle["attempted"] == 1 and cycle["mae_kwh"] == 10.0
    assert report["gates"]["A4"]["paired_n"] == 1
    assert report["gates"]["A4"]["status"] == "fail"
    later = [row for row in report["M6_lead_diagnostics"] if row["model"] == "cycle"]
    assert [row["lead_min"] for row in later] == [1440, 2880]
    assert all(row["mae_kwh"] == 0 for row in later)
