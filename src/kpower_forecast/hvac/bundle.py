"""Read an explicitly inventoried HVAC evaluation bundle without filling its data.

This module validates source evidence. Loading a bundle does not qualify its
observations, fit a model or authorize planning with its contents.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class BundleArtifact(BaseModel):
    """An immutable file identity recorded by the bundle exporter."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["parquet", "json"]


class BundleManifest(BaseModel):
    """Versioned inventory and provenance, separate from training eligibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    bundle_version: Literal[1] = 1
    exported_at: AwareDatetime
    start: AwareDatetime
    end: AwareDatetime
    controller_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    timezone: str
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    ems_commit: str = Field(min_length=7)
    ingest_lag_min: int | None = Field(default=None, ge=0)
    training_ready: bool = False
    files: dict[str, BundleArtifact]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @model_validator(mode="after")
    def _ordered(self) -> BundleManifest:
        if self.end <= self.start:
            raise ValueError("bundle end must follow start")
        if self.training_ready and self.ingest_lag_min is None:
            raise ValueError("training-ready bundle requires measured ingestion lag")
        return self


def _utc_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f"{name}: timestamps must be timezone-aware")
    if (
        str(index.tz) != "UTC"
        or not index.is_monotonic_increasing
        or not index.is_unique
    ):
        raise ValueError(f"{name}: timestamps must be sorted, unique UTC")
    return index


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _validate_minutes(frame: pd.DataFrame, manifest: BundleManifest) -> None:
    index = _utc_index(frame.index, "minutes")
    if "power_w" not in frame or not index.floor("min").equals(index):
        raise ValueError("minutes require power_w at minute starts")
    if len(index) and (
        (index < pd.Timestamp(manifest.start)).any()
        or (index >= pd.Timestamp(manifest.end)).any()
    ):
        raise ValueError("minutes fall outside bundle span")
    values = pd.to_numeric(frame.power_w, errors="raise")
    if values.isin([float("inf"), -float("inf")]).any() or (values < 0).any():
        raise ValueError("invalid electrical power")


@dataclass(frozen=True)
class HvacEvaluationBundle:
    """Validated private-source inventory usable by the evaluation API.

    Tables preserve observed gaps, provenance and extra quality columns. Numeric
    model code is deliberately not imported by the loader.
    """

    manifest: BundleManifest
    tables: Mapping[str, pd.DataFrame]
    documents: Mapping[str, JsonValue]

    @classmethod
    def load(cls, directory: Path) -> HvacEvaluationBundle:
        """Read only files listed in a manifest and verify their exact identities.

        Args:
            directory: Export directory containing manifest.json and its files.

        Returns:
            Validated tables/documents with missing observations preserved.

        Raises:
            ValueError: If inventory, hashes or the minute contract are invalid.
            OSError: If an inventoried file cannot be read.
        """
        root = directory.resolve(strict=True)
        manifest = BundleManifest.model_validate(
            json.loads(
                (root / "manifest.json").read_bytes(), parse_constant=_reject_constant
            )
        )
        tables: dict[str, pd.DataFrame] = {}
        documents: dict[str, JsonValue] = {}
        for name, spec in manifest.files.items():
            if Path(name).name != name or name in {".", "..", "manifest.json"}:
                raise ValueError("artifact name must be a local filename")
            path = (root / name).resolve(strict=True)
            if path.parent != root:
                raise ValueError("artifact escapes bundle directory")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != spec.sha256:
                raise ValueError(f"artifact hash mismatch: {name}")
            if spec.format == "parquet":
                tables[name] = pd.read_parquet(io.BytesIO(data))
            else:
                documents[name] = json.loads(data, parse_constant=_reject_constant)
        if "minutes.parquet" not in tables or "capabilities.json" not in documents:
            raise ValueError("bundle requires minutes and capability evidence")
        if not isinstance(documents["capabilities.json"], dict):
            raise ValueError("capability evidence must be a JSON object")
        _validate_minutes(tables["minutes.parquet"], manifest)
        return cls(manifest, MappingProxyType(tables), MappingProxyType(documents))

    def available_at(self, origin: datetime) -> pd.DataFrame:
        """Return an isolated observed-history slice after the availability boundary.

        Args:
            origin: Aware UTC origin of a fit or prediction.

        Returns:
            Copy of minute rows whose end plus measured lag is at/before origin.

        Raises:
            ValueError: If the origin is not UTC or ingestion lag is unmeasured.
        """
        offset = origin.utcoffset()
        if origin.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("origin must be aware UTC")
        if self.manifest.ingest_lag_min is None:
            raise ValueError("ingestion lag is not measured")
        frame = self.tables["minutes.parquet"]
        cutoff = pd.Timestamp(origin) - pd.Timedelta(
            minutes=1 + self.manifest.ingest_lag_min
        )
        return frame.loc[frame.index <= cutoff].copy(deep=True)
