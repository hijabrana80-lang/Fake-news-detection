"""Model lifecycle: locate, load and cache the trained classifier.

The Flask app and the CLI both go through this module so the model is only
ever loaded from disk once.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from .model import NewsClassifier

__all__ = ["ModelRegistry", "get_registry", "model_path", "MODELS_DIR"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODELS_DIR, "news_classifier.joblib")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")


def model_path() -> str:
    return MODEL_FILE


class ModelRegistry:
    """Thread-safe lazy loader for the trained model."""

    def __init__(self, path: str = MODEL_FILE):
        self.path = path
        self._model: Optional[NewsClassifier] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)

    @property
    def error(self) -> Optional[str]:
        return self._error

    def load(self, force: bool = False) -> Optional[NewsClassifier]:
        with self._lock:
            if self._model is not None and not force:
                return self._model
            if not os.path.exists(self.path):
                self._error = (
                    "Model file not found. Train it first with `python scripts/train.py`."
                )
                self._model = None
                return None
            try:
                self._model = NewsClassifier.load(self.path)
                self._error = None
            except Exception as exc:
                self._model = None
                self._error = f"Could not load model: {exc}"
            return self._model

    def get(self) -> NewsClassifier:
        model = self.load()
        if model is None:
            raise RuntimeError(self._error or "Model is not available.")
        return model

    def unload(self) -> None:
        with self._lock:
            self._model = None


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
