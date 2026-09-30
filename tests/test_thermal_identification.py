"""Offline chronology, physical rejection and confidence invariants."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from kpower_forecast.thermal.identification import (
    MatrixBundle,
    MatrixManifest,
    SensorConfidence,
    SourceTransition,
    confidence_weights,
    matrix_report,
    publish_report,
)


@pytest.fixture
def bundle() -> MatrixBundle:
    """Freeze four weeks of real synthetic hourly endpoints and on/off input."""
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    instant = origin
    indoor = 19.0
    rows = []
    for index in range(28 * 24):
        outside = 3 + 7 * np.sin(index / 37)
        power = 1800 if index % 18 < 9 else 0
        decay = np.exp(-1 / 22)
        end_indoor = decay * indoor + (1 - decay) * (outside + 4 * power / 1000 + 10)
        rows.append(
            SourceTransition(
                start_at=instant,
                end_at=instant + timedelta(hours=1),
                indoor_start_c=indoor,
                indoor_end_c=float(end_indoor),
                outdoor_mean_c=float(outside),
                hvac_electric_mean_w=power,
                hvac_coverage_ratio=1,
                source_authority_id="sensor-a",
            )
        )
        instant, indoor = rows[-1].end_at, end_indoor
    manifest = MatrixManifest(
        experiment_id="synthetic-hourly-v1",
        model_id="zone-test",
        authority_epoch=origin,
        export_start_at=origin,
        fit_cutoff=origin + timedelta(days=21),
        evaluation_start=origin + timedelta(days=21, hours=6),
        cutoff=instant,
        input_provenance="synthetic exact RC hourly exposure",
    )
    return MatrixBundle(
        manifest=manifest, series={m: rows for m in (5, 10, 15, 30, 60)}
    )


def test_matrix_recovers_physics_without_fabricated_half_hour_targets(
    bundle: MatrixBundle,
) -> None:
    """All cells retain limits and use the same genuine evaluation endpoints."""
    report = matrix_report(bundle)
    cells = report["cells"]
    assert len(cells) == 30
    assert {cell["status"] for cell in cells} == {"fitted"}
    assert cells[0]["candidate"]["tau_hours"] == pytest.approx(22, rel=0.15)
    assert cells[0]["candidate"]["gain_c_per_kw"] == pytest.approx(4, rel=0.15)
    assert cells[0]["candidate"]["boundary"] is None
    assert {cell["calibration_metrics"]["30min"]["matched"] for cell in cells} == {0}
    assert {cell["calibration_metrics"]["60min"]["matched"] for cell in cells} == {162}
    assert cells[0]["calibration_metrics"]["240min"]["improvement_c"] > 0
    assert report["empirically_reliable"] is False
    assert (
        len(
            [
                edge
                for edge in report["neighbors"]
                if edge["first"][0] == 5 and edge["second"][0] == 10
            ]
        )
        == 6
    )


def test_external_outcomes_cannot_change_fit_and_missing_windows_are_unavailable(
    bundle: MatrixBundle,
) -> None:
    """No evaluation leakage or relabeling short history as a long window."""
    original = matrix_report(bundle)
    altered = bundle.model_copy(
        update={
            "series": {
                minutes: [
                    (
                        row.model_copy(update={"indoor_end_c": row.indoor_end_c + 4})
                        if row.start_at >= bundle.manifest.evaluation_start
                        else row
                    )
                    for row in rows
                ]
                for minutes, rows in bundle.series.items()
            }
        }
    )
    changed = matrix_report(altered)
    assert [cell["candidate"] for cell in original["cells"]] == [
        cell["candidate"] for cell in changed["cells"]
    ]
    assert (
        original["cells"][0]["calibration_metrics"]
        != changed["cells"][0]["calibration_metrics"]
    )
    shorter = bundle.model_copy(
        update={
            "manifest": bundle.manifest.model_copy(
                update={
                    "export_start_at": bundle.manifest.fit_cutoff - timedelta(days=8),
                }
            ),
            "series": {
                minutes: rows[13 * 24 :] for minutes, rows in bundle.series.items()
            },
        }
    )
    short_report = matrix_report(MatrixBundle.model_validate(shorter.model_dump()))
    assert {cell["status"] for cell in short_report["cells"] if cell["days"] > 7} == {
        "unavailable_history"
    }
    assert {cell["status"] for cell in short_report["cells"] if cell["days"] == 7} == {
        "fitted"
    }


def test_confidence_is_source_specific_bounded_and_elapsed_aware(
    bundle: MatrixBundle,
) -> None:
    """Coarse resolution reduces confidence; unknown accuracy stays assumed."""
    first, second = bundle.series[5][:2]
    rows = [first, second.model_copy(update={"source_authority_id": "coarse"})]
    manifest = bundle.manifest.model_copy(
        update={
            "confidence": {
                "coarse": SensorConfidence(
                    provenance="documented 0.2C quantization", resolution_c=0.2
                ),
            }
        }
    )
    weighted = confidence_weights(rows, manifest)
    assert weighted[0] > weighted[1]
    assert np.mean(weighted) == pytest.approx(1)
    capped = manifest.model_copy(update={"max_confidence_ratio": 1.1})
    bounded = confidence_weights(rows, capped)
    assert bounded[0] / bounded[1] == pytest.approx(1.1)
    longer = second.model_copy(update={"end_at": second.start_at + timedelta(hours=2)})
    assert confidence_weights([first, longer], bundle.manifest)[1] == pytest.approx(
        4 / 3
    )


@pytest.mark.parametrize("change", ["embargo", "overlap", "coverage"])
def test_bundle_rejects_unadmitted_evidence(bundle: MatrixBundle, change: str) -> None:
    """Chronology and exact-coverage errors fail at the private input boundary."""
    payload = bundle.model_dump()
    if change == "embargo":
        payload["manifest"]["evaluation_start"] = bundle.manifest.fit_cutoff
    elif change == "overlap":
        payload["series"][5][1]["start_at"] = bundle.series[5][0].start_at
    else:
        payload["series"][5][0]["hvac_coverage_ratio"] = 0.5
    with pytest.raises(ValidationError):
        MatrixBundle.model_validate(payload)


def test_offset_times_serialize_canonically_in_utc(bundle: MatrixBundle) -> None:
    """Offset representations retain identity but never enter window arithmetic."""
    payload = bundle.model_dump()
    offset = timezone(timedelta(hours=2))
    for key in (
        "authority_epoch",
        "export_start_at",
        "fit_cutoff",
        "evaluation_start",
        "cutoff",
    ):
        payload["manifest"][key] = payload["manifest"][key].astimezone(offset)
    for rows in payload["series"].values():
        for row in rows:
            row["start_at"] = row["start_at"].astimezone(offset)
            row["end_at"] = row["end_at"].astimezone(offset)
    normalized = MatrixBundle.model_validate(payload)
    assert normalized.model_dump_json() == bundle.model_dump_json()
    assert normalized.manifest.fit_cutoff.tzinfo is timezone.utc
    assert normalized.series[5][0].start_at.tzinfo is timezone.utc


def test_interrupted_report_is_retryable_and_existing_data_is_preserved(
    tmp_path: Path,
) -> None:
    """Incomplete staging is never the final report; publication never clobbers."""
    output = tmp_path / "private-report.json"
    contents = '{"complete": true}\n'
    with patch(
        "kpower_forecast.thermal.identification.os.link",
        side_effect=OSError("interrupted"),
    ):
        with pytest.raises(OSError, match="interrupted"):
            publish_report(output, contents)
    assert not output.exists()
    publish_report(output, contents)
    publish_report(output, contents)
    assert output.read_text() == contents
    with pytest.raises(ValueError, match="different contents"):
        publish_report(output, '{"different": true}\n')
    assert output.read_text() == contents


@pytest.mark.parametrize("tau,gain", [(300, 4), (22, -4)])
def test_raw_boundary_and_negative_gain_are_diagnostic_only(
    bundle: MatrixBundle, tau: float, gain: float
) -> None:
    """Rejected raw optima remain visible without manufacturing acceptance."""
    rows = [
        row.model_copy(
            update={
                "indoor_end_c": float(
                    np.exp(-1 / tau) * row.indoor_start_c
                    + (1 - np.exp(-1 / tau))
                    * (row.outdoor_mean_c + gain * row.hvac_electric_mean_w / 1000 + 10)
                ),
            }
        )
        for row in bundle.series[5]
    ]
    altered = MatrixBundle(
        manifest=bundle.manifest, series={m: rows for m in (5, 10, 15, 30, 60)}
    )
    cell = matrix_report(altered)["cells"][0]
    assert cell["status"] == "implausible_parameters"
    assert cell["calibration_metrics"] == {}
    if tau > 240:
        assert cell["candidate"]["boundary"] == "upper"
        assert cell["candidate"]["tau_hours"] == 240
    else:
        assert cell["candidate"]["gain_c_per_kw"] < 0
