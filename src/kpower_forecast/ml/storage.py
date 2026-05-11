"""Storage helpers for optional ML model artifacts."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class MLModelManifest(BaseModel):
    """JSON-serializable manifest describing ML model artifacts."""

    model_id: str
    backend_type: str
    target_type: str
    interval_levels: list[int]
    feature_columns: list[str]
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    conformal_quantiles: dict[str, float] = Field(default_factory=dict)
    weather_bias_source: Optional[str] = None
    training_start: Optional[str] = None
    training_end: Optional[str] = None
    package_version: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MLModelStorage:
    """Persist ML manifests and supporting data under a storage directory."""

    def __init__(self, storage_path: str, model_id: str):
        self.storage_path = Path(storage_path)
        self.model_id = model_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def artifact_dir(self) -> Path:
        """Return the artifact directory for this model."""
        return self.storage_path / f"{self.model_id}_ml"

    @property
    def manifest_path(self) -> Path:
        """Return the active manifest path."""
        return self.storage_path / f"{self.model_id}_ml_manifest.json"

    def save_manifest(self, manifest: MLModelManifest) -> None:
        """Persist a manifest atomically enough for local single-process use."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest.model_dump(mode="json"), file, indent=2, sort_keys=True)

    def load_manifest(self) -> Optional[MLModelManifest]:
        """Load the active manifest, returning None when absent."""
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return MLModelManifest.model_validate(payload)

    def save_training_frame(self, df: pd.DataFrame) -> None:
        """Persist prepared ML training data for later update/debug workflows."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.artifact_dir / "training.parquet", index=False)
