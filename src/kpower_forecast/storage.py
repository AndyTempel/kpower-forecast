import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

logger = logging.getLogger(__name__)


class ModelStorage:
    """
    Handles persistence of Prophet models and associated metadata.
    """

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_path.exists():
            logger.info(f"Creating storage directory: {self.storage_path}")
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_id: str) -> Path:
        return self.storage_path / f"{model_id}.json"

    def _get_data_path(self, model_id: str) -> Path:
        return self.storage_path / f"{model_id}_data.parquet"

    def _get_forecast_archive_path(self, model_id: str, forecast_model: str) -> Path:
        safe_forecast_model = forecast_model.replace("/", "_")
        return self.storage_path / f"{model_id}_{safe_forecast_model}_forecast.parquet"

    def _get_weather_calibration_path(self, model_id: str, forecast_model: str) -> Path:
        safe_forecast_model = forecast_model.replace("/", "_")
        return self.storage_path / f"{model_id}_{safe_forecast_model}_weather.json"

    def exists(self, model_id: str) -> bool:
        """
        Checks if a model with the given ID exists in storage.
        """
        return self._get_model_path(model_id).exists()

    def has_training_data(self, model_id: str) -> bool:
        """Returns True if a training data parquet exists for the given model_id."""
        return self._get_data_path(model_id).exists()

    def save_model(
        self, model: Prophet, model_id: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Serializes and saves the Prophet model and metadata to disk.
        """
        file_path = self._get_model_path(model_id)
        try:
            logger.info(f"Saving model {model_id} to {file_path}")
            data = {
                "model": model_to_json(model),
                "metadata": metadata or {},
            }
            with open(file_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise

    def load_model(self, model_id: str) -> Optional[Tuple[Prophet, Dict[str, Any]]]:
        """
        Loads the Prophet model and metadata from disk.
        Returns (Prophet, metadata) or None if not found.
        """
        file_path = self._get_model_path(model_id)
        if not file_path.exists():
            logger.info(f"No saved model found for {model_id} at {file_path}")
            return None

        try:
            logger.info(f"Loading model {model_id} from {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)

            # Backwards compatibility for models saved without metadata wrapper
            if "model" in data and "metadata" in data:
                return model_from_json(data["model"]), data["metadata"]
            else:
                # Old format: the whole file is the model JSON
                return model_from_json(data), {}

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def save_training_data(self, model_id: str, df: pd.DataFrame) -> None:
        """Saves the prepared training DataFrame (with weather cols) as parquet."""
        path = self._get_data_path(model_id)
        try:
            df.to_parquet(path, index=False)
            logger.info(f"Saved training data for {model_id} to {path}")
        except Exception as e:
            logger.error(f"Failed to save training data for {model_id}: {e}")
            raise

    def load_training_data(self, model_id: str) -> Optional[pd.DataFrame]:
        """Loads the prepared training DataFrame from parquet.

        Returns None if not found.
        """
        path = self._get_data_path(model_id)
        if not path.exists():
            return None
        try:
            logger.info(f"Loading training data for {model_id} from {path}")
            return pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Failed to load training data for {model_id}: {e}")
            raise

    def save_forecast_archive(
        self, model_id: str, forecast_model: str, df: pd.DataFrame
    ) -> None:
        """Merge and persist timestamped forecast archive rows.

        Args:
            model_id: Unique model identifier.
            forecast_model: Weather model identifier.
            df: Forecast archive dataframe containing at least ``ds``.

        Returns:
            None.

        Raises:
            ValueError: If ``df`` is missing the ``ds`` column.
        """
        if "ds" not in df.columns:
            raise ValueError("forecast archive must contain a 'ds' column")

        path = self._get_forecast_archive_path(model_id, forecast_model)
        archive = df.copy()
        archive["ds"] = pd.to_datetime(archive["ds"], utc=True)

        if path.exists():
            existing = pd.read_parquet(path)
            existing["ds"] = pd.to_datetime(existing["ds"], utc=True)
            archive = pd.concat([existing, archive], ignore_index=True)

        archive = (
            archive.drop_duplicates(subset=["ds"], keep="last")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        archive.to_parquet(path, index=False)
        logger.info("Saved forecast archive for %s to %s", model_id, path)

    def load_forecast_archive(
        self, model_id: str, forecast_model: str
    ) -> Optional[pd.DataFrame]:
        """Load forecast archive rows for a model and weather model.

        Args:
            model_id: Unique model identifier.
            forecast_model: Weather model identifier.

        Returns:
            DataFrame of forecast archive rows, or None when unavailable.
        """
        path = self._get_forecast_archive_path(model_id, forecast_model)
        if not path.exists():
            return None
        archive = pd.read_parquet(path)
        archive["ds"] = pd.to_datetime(archive["ds"], utc=True)
        return archive

    def save_weather_calibration(
        self, model_id: str, forecast_model: str, metadata: Dict[str, Any]
    ) -> None:
        """Persist adaptive weather calibration metadata as JSON.

        Args:
            model_id: Unique model identifier.
            forecast_model: Weather model identifier.
            metadata: JSON-serializable calibration metadata.

        Returns:
            None.
        """
        path = self._get_weather_calibration_path(model_id, forecast_model)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(metadata, file)
        logger.info("Saved weather calibration for %s to %s", model_id, path)

    def load_weather_calibration(
        self, model_id: str, forecast_model: str
    ) -> Optional[Dict[str, Any]]:
        """Load adaptive weather calibration metadata.

        Args:
            model_id: Unique model identifier.
            forecast_model: Weather model identifier.

        Returns:
            Calibration metadata, or None when unavailable.
        """
        path = self._get_weather_calibration_path(model_id, forecast_model)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Weather calibration payload must be a JSON object.")

        return cast(Dict[str, Any], payload)

    def delete_model(self, model_id: str) -> None:
        """Removes the model JSON and training data parquet for a given model_id."""
        for path in [self._get_model_path(model_id), self._get_data_path(model_id)]:
            if path.exists():
                path.unlink()
                logger.info(f"Deleted {path}")
