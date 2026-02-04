import json
import logging
from pathlib import Path
from typing import Optional

from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

logger = logging.getLogger(__name__)


class ModelStorage:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_path.exists():
            logger.info(f"Creating storage directory: {self.storage_path}")
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_id: str) -> Path:
        return self.storage_path / f"{model_id}.json"

    def save_model(self, model: Prophet, model_id: str):
        """
        Serializes and saves the Prophet model to disk.
        """
        file_path = self._get_model_path(model_id)
        try:
            logger.info(f"Saving model {model_id} to {file_path}")
            with open(file_path, "w") as f:
                json.dump(model_to_json(model), f)
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise

    def load_model(self, model_id: str) -> Optional[Prophet]:
        """
        Loads the Prophet model from disk if it exists.
        Returns None if not found.
        """
        file_path = self._get_model_path(model_id)
        if not file_path.exists():
            logger.info(f"No saved model found for {model_id} at {file_path}")
            return None

        try:
            logger.info(f"Loading model {model_id} from {file_path}")
            with open(file_path, "r") as f:
                return model_from_json(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            # If load fails, we might want to return None to trigger retraining,
            # or raise to alert the user.
            # Given "production-grade", maybe explicit failure is safer
            # than silent fallback?
            # But the prompt says "If a model exists, load it...
            # If no model exists ... train".
            # If it exists but is corrupt, maybe we should raise.
            raise
