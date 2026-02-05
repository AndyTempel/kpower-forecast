import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

    def exists(self, model_id: str) -> bool:
        """
        Checks if a model with the given ID exists in storage.
        """
        return self._get_model_path(model_id).exists()

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
