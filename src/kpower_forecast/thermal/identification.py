"""Offline RC matrix and calibration replay; never writes operational artifacts.

Input adapters own source/receipt/quality and exact exposure validation. This
module consumes only their admitted real endpoints and pre-exported weather.
"""

import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .fitting import FitCandidate, fit_candidate
from .model import (
    KPowerThermalForecast,
    ThermalModelConfig,
    ThermalTrainingTransition,
    _utc,
)


class Evidence(BaseModel):
    """Immutable input with unknown fields rejected at the offline boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SensorConfidence(Evidence):
    """Documented source confidence; precision is not a claim of accuracy."""

    provenance: str = Field(min_length=1)
    resolution_c: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    noise_sigma_c: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class SourceTransition(ThermalTrainingTransition):
    """Adapter-verified transition with the same source at both real endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_authority_id: str = Field(min_length=1)

    @field_validator("start_at", "end_at")
    @classmethod
    def canonical_time(cls, value: datetime) -> datetime:
        """Keep real source instants canonical in UTC, without changing identity."""
        return _utc(value)


class MatrixManifest(Evidence):
    """Frozen calibration experiment; final holdout and promotion are excluded."""

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    authority_epoch: datetime
    export_start_at: datetime
    fit_cutoff: datetime
    evaluation_start: datetime
    cutoff: datetime
    input_provenance: str = Field(min_length=1)
    noise_floor_c: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    max_confidence_ratio: float = Field(default=10.0, ge=1, allow_inf_nan=False)
    target_tolerance_minutes: float = Field(default=7.5, gt=0, le=7.5)
    confidence: dict[str, SensorConfidence] = Field(default_factory=dict)

    @field_validator(
        "authority_epoch", "export_start_at", "fit_cutoff", "evaluation_start", "cutoff"
    )
    @classmethod
    def canonical_time(cls, value: datetime) -> datetime:
        """Normalize aware experiment instants before window arithmetic."""
        return _utc(value)

    @model_validator(mode="after")
    def validate_times(self) -> "MatrixManifest":
        """Require UTC-aware chronology and the six-hour transition embargo."""
        for at in (
            self.authority_epoch,
            self.export_start_at,
            self.fit_cutoff,
            self.evaluation_start,
            self.cutoff,
        ):
            if at.tzinfo is None or at.utcoffset() is None:
                raise ValueError("experiment timestamps must be timezone-aware")
        if not self.export_start_at < self.fit_cutoff:
            raise ValueError("export must begin before fit cutoff")
        if self.evaluation_start < self.fit_cutoff + timedelta(hours=6):
            raise ValueError("evaluation needs a six-hour transition embargo")
        if self.evaluation_start >= self.cutoff:
            raise ValueError("evaluation must begin before the frozen cutoff")
        return self


class MatrixBundle(Evidence):
    """Per-resolution exports; five-minute evidence supplies common test targets."""

    manifest: MatrixManifest
    series: dict[int, list[SourceTransition]]

    @model_validator(mode="after")
    def validate_series(self) -> "MatrixBundle":
        """Reject unsupported intervals, overlaps and unadmitted exposure."""
        if set(self.series) != {5, 10, 15, 30, 60}:
            raise ValueError(
                "export all five endpoint selections, including empty ones"
            )
        for minutes, rows in self.series.items():
            previous: SourceTransition | None = None
            for row in rows:
                if (
                    row.start_at
                    < max(self.manifest.authority_epoch, self.manifest.export_start_at)
                    or row.end_at > self.manifest.cutoff
                    or row.elapsed_hours * 60 < minutes
                    or row.elapsed_hours > 6
                    or row.hvac_coverage_ratio
                    < ThermalModelConfig().min_hvac_coverage_ratio
                ):
                    raise ValueError("transition is outside admitted export coverage")
                if previous is not None and row.start_at < previous.end_at:
                    raise ValueError("transitions must be ordered and nonoverlapping")
                previous = row
        return self


class MatrixCell(Evidence):
    """Every requested fit, including unavailable and rejected configurations."""

    minutes: int
    days: int
    weighting: Literal["unweighted", "confidence_elapsed"]
    status: str
    requested_start_at: datetime
    available_span_hours: float
    covered_hours: float = 0
    transition_count: int = 0
    candidate: FitCandidate | None = None
    internal_holdout_reason: str | None = None
    fit_reliable_on_holdout: bool = False
    calibration_metrics: dict[str, dict[str, float | int | None]] = Field(
        default_factory=dict
    )


def confidence_weights(
    rows: list[SourceTransition], manifest: MatrixManifest
) -> np.ndarray:
    """Bound source confidence and balance actual elapsed exposure.

    Args:
        rows: Admitted real transitions.
        manifest: Frozen noise assumptions; unknown noise uses the named floor.

    Returns:
        Positive normalized weights. Endpoint covariance uses a conservative
        bound; cycle weighting and initial-regressor error remain unresolved.
    """
    confidence = []
    for row in rows:
        sensor = manifest.confidence.get(row.source_authority_id)
        variance = manifest.noise_floor_c**2
        if sensor is not None:
            variance += (sensor.resolution_c or 0) ** 2 / 12
            variance += (sensor.noise_sigma_c or 0) ** 2
        # Both endpoints contribute, with no independence assumption.
        if (
            not math.isfinite(variance)
            or variance <= 0
            or not math.isfinite(4 * variance)
        ):
            raise ValueError("sensor confidence variance is outside numeric range")
        precision = 1 / (4 * variance)
        if not math.isfinite(precision) or precision <= 0:
            raise ValueError("sensor confidence precision is outside numeric range")
        confidence.append(precision)
    values = np.array(confidence)
    if not len(values):
        return values
    values /= np.max(values)
    values = np.maximum(values, 1 / manifest.max_confidence_ratio)
    values *= np.array([row.elapsed_hours for row in rows])
    return cast(np.ndarray, values / np.mean(values))


class _DiagnosticModel(KPowerThermalForecast):
    """Capture the raw optimum while retaining the runtime's acceptance gates."""

    def __init__(
        self,
        rows: list[SourceTransition],
        manifest: MatrixManifest,
        minutes: int,
        weighting: str,
    ) -> None:
        super().__init__(
            model_id=manifest.model_id,
            authority_fingerprint=manifest.experiment_id,
            storage_path=Path("."),
            config=ThermalModelConfig(min_transition_minutes=minutes),
        )
        self.candidate: FitCandidate | None = None
        self.weights = (
            confidence_weights(rows, manifest) if weighting != "unweighted" else None
        )

    def _fit(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        outdoors: np.ndarray,
        power_kw: np.ndarray,
        hours: np.ndarray,
    ) -> tuple[float, float, float] | None:
        self.candidate = fit_candidate(
            starts,
            ends,
            outdoors,
            power_kw,
            hours,
            min_tau=self.config.min_time_constant_hours,
            max_tau=self.config.max_time_constant_hours,
            grid_points=self.config.fit_grid_points,
            weights=self.weights[: len(starts)] if self.weights is not None else None,
        )
        return self._accepted_parameters(self.candidate)


def _calibration_metrics(
    rows: list[SourceTransition],
    candidate: FitCandidate,
    tolerance_minutes: float,
) -> dict[str, dict[str, float | int | None]]:
    horizons = (30, 60, 120, 240)
    errors: dict[int, list[float]] = {h: [] for h in horizons}
    baselines: dict[int, list[float]] = {h: [] for h in horizons}
    offsets: dict[int, list[float]] = {h: [] for h in horizons}
    blocks: dict[int, set[str]] = {h: set() for h in horizons}
    for index, origin_row in enumerate(rows):
        state = origin_row.indoor_start_c
        previous = origin_row
        targets: dict[int, tuple[float, float, float]] = {}
        for row in rows[index:]:
            if row is not origin_row and (
                row.start_at != previous.end_at
                or row.indoor_start_c != previous.indoor_end_c
                or row.source_authority_id != previous.source_authority_id
            ):
                break
            state = float(
                KPowerThermalForecast._step(
                    np.array([state]),
                    np.array([row.outdoor_mean_c]),
                    np.array([row.hvac_electric_mean_w / 1000]),
                    np.array([row.elapsed_hours]),
                    candidate.parameters,
                )[0]
            )
            elapsed = (row.end_at - origin_row.start_at).total_seconds() / 60
            for horizon in horizons:
                offset = elapsed - horizon
                if abs(offset) <= tolerance_minutes and (
                    horizon not in targets or abs(offset) < abs(targets[horizon][2])
                ):
                    targets[horizon] = (
                        state - row.indoor_end_c,
                        origin_row.indoor_start_c - row.indoor_end_c,
                        offset,
                    )
            previous = row
            if elapsed > max(horizons) + tolerance_minutes:
                break
        for horizon, (error, baseline, offset) in targets.items():
            errors[horizon].append(error)
            baselines[horizon].append(baseline)
            offsets[horizon].append(abs(offset))
            blocks[horizon].add(
                origin_row.start_at.astimezone(timezone.utc).date().isoformat()
            )
    result: dict[str, dict[str, float | int | None]] = {}
    for horizon in horizons:
        residuals, baseline_errors = (
            np.array(errors[horizon]),
            np.array(baselines[horizon]),
        )
        matched = len(residuals)
        mae = float(np.mean(np.abs(residuals))) if matched else None
        persistence = float(np.mean(np.abs(baseline_errors))) if matched else None
        result[f"{horizon}min"] = {
            "matched": matched,
            "unmatched": len(rows) - matched,
            "utc_day_blocks": len(blocks[horizon]),
            "mae_c": mae,
            "bias_c": float(np.mean(residuals)) if matched else None,
            "rmse_c": float(np.sqrt(np.mean(residuals**2))) if matched else None,
            "persistence_mae_c": persistence,
            "improvement_c": (
                persistence - mae
                if persistence is not None and mae is not None
                else None
            ),
            "max_target_offset_minutes": max(offsets[horizon]) if matched else None,
        }
    return result


def matrix_report(bundle: MatrixBundle) -> dict[str, object]:
    """Fit the fixed matrix and replay common, untouched calibration outcomes.

    Args:
        bundle: Frozen, adapter-validated true observations and observed inputs.

    Returns:
        JSON-compatible report with all cells, adjacent comparisons and explicit
        limitations. No model artifacts, commands or network requests are made.
    """
    manifest = bundle.manifest
    evaluation = [
        row for row in bundle.series[5] if row.start_at >= manifest.evaluation_start
    ]
    cells: list[MatrixCell] = []
    for minutes in (5, 10, 15, 30, 60):
        for days in (7, 14, 21):
            start = manifest.fit_cutoff - timedelta(days=days)
            available_start = max(manifest.authority_epoch, manifest.export_start_at)
            rows = [
                row
                for row in bundle.series[minutes]
                if row.start_at >= start and row.end_at <= manifest.fit_cutoff
            ]
            for weighting in ("unweighted", "confidence_elapsed"):
                cell = MatrixCell(
                    minutes=minutes,
                    days=days,
                    weighting=weighting,
                    requested_start_at=start,
                    available_span_hours=max(
                        0,
                        (manifest.fit_cutoff - available_start).total_seconds() / 3600,
                    ),
                    covered_hours=sum(row.elapsed_hours for row in rows),
                    transition_count=len(rows),
                    status="unavailable_history",
                )
                if available_start <= start:
                    model = _DiagnosticModel(rows, manifest, minutes, weighting)
                    training_rows: list[ThermalTrainingTransition] = list(rows)
                    diagnostics = model.train(training_rows)
                    cell = cell.model_copy(
                        update={
                            "status": (
                                "fitted"
                                if diagnostics.fitted
                                else diagnostics.unreliable_reason
                            ),
                            "candidate": model.candidate,
                            "fit_reliable_on_holdout": diagnostics.reliable_on_holdout,
                            "internal_holdout_reason": diagnostics.unreliable_reason,
                            "calibration_metrics": (
                                _calibration_metrics(
                                    evaluation,
                                    model.candidate,
                                    manifest.target_tolerance_minutes,
                                )
                                if diagnostics.fitted and model.candidate is not None
                                else {}
                            ),
                        }
                    )
                cells.append(cell)
    by_key = {(cell.minutes, cell.days, cell.weighting): cell for cell in cells}
    neighbors: list[dict[str, object]] = []
    for cell in cells:
        next_minutes = {5: 10, 10: 15, 15: 30, 30: 60}.get(cell.minutes)
        next_days = {7: 14, 14: 21}.get(cell.days)
        for other_key in (
            (next_minutes, cell.days, cell.weighting),
            (cell.minutes, next_days, cell.weighting),
        ):
            if other_key[0] is None or other_key[1] is None:
                continue
            other = by_key.get((other_key[0], other_key[1], other_key[2]))
            if other is None:
                continue
            first, second = cell.candidate, other.candidate
            neighbors.append(
                {
                    "first": [cell.minutes, cell.days, cell.weighting],
                    "second": [other.minutes, other.days, other.weighting],
                    "statuses": [cell.status, other.status],
                    "gain_sign_same": (
                        (
                            math.copysign(1, first.gain_c_per_kw)
                            == math.copysign(1, second.gain_c_per_kw)
                        )
                        if first and second
                        else None
                    ),
                    "tau_difference_hours": (
                        abs(first.tau_hours - second.tau_hours)
                        if first and second
                        else None
                    ),
                    "gain_difference_c_per_kw": (
                        abs(first.gain_c_per_kw - second.gain_c_per_kw)
                        if first and second
                        else None
                    ),
                    "offset_difference_c": (
                        abs(first.offset_c - second.offset_c)
                        if first and second
                        else None
                    ),
                }
            )
    return {
        "schema_version": 1,
        "manifest": manifest.model_dump(mode="json"),
        "input_sha256": hashlib.sha256(bundle.model_dump_json().encode()).hexdigest(),
        "evaluation_layer": "observed_drive_response_calibration",
        "cells": [cell.model_dump(mode="json") for cell in cells],
        "neighbors": neighbors,
        "empirically_reliable": False,
        "limitations": [
            "No final holdout selection or operational forecast replay.",
            "UTC day blocks and neighboring fits are not independent cycles.",
            "Noise floors are explicit assumptions, not measured sensor accuracy.",
            (
                "Cycle balancing, excitation calibration, disturbance classification "
                "and interval calibration remain unresolved."
            ),
            (
                "Weighted endpoint confidence does not solve "
                "initial-regressor measurement error."
            ),
        ],
    }


def publish_report(output_path: Path, contents: str) -> None:
    """Atomically publish a complete private report without replacing user data.

    Args:
        output_path: Final path in an existing directory.
        contents: Complete serialized report.

    Raises:
        ValueError: If a different report already exists.
        OSError: If staging or atomic no-clobber publication fails.
    """
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != contents:
            raise ValueError("output exists with different contents")
        return
    # The unique sibling is private scratch created by this invocation. A
    # killed process can leave scratch, but never an incomplete final report.
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete_on_close=False,
    ) as staging:
        staging.write(contents)
        staging.flush()
        os.fsync(staging.fileno())
        staging.close()
        try:
            os.link(staging.name, output_path)
        except FileExistsError:
            if output_path.read_text(encoding="utf-8") != contents:
                raise ValueError("output exists with different contents") from None


def main() -> None:
    """Run the optional Typer CLI with an offline bundle and exclusive output."""
    import typer
    from rich.console import Console

    app = typer.Typer(add_completion=False)

    @app.command()
    def report(
        bundle_path: Annotated[
            Path, typer.Argument(help="Frozen private evidence bundle")
        ],
        output_path: Annotated[
            Path, typer.Argument(help="Exclusive private report path")
        ],
    ) -> None:
        """Write a private JSON report; never overwrite a different report."""
        bundle = MatrixBundle.model_validate_json(bundle_path.read_text())
        contents = json.dumps(matrix_report(bundle), indent=2, allow_nan=False) + "\n"
        try:
            publish_report(output_path, contents)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        Console().print(f"Offline calibration report: {output_path}")

    app()


if __name__ == "__main__":
    main()
