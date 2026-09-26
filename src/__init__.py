"""Fake news detection package."""

from .preprocessing import clean_text, tokenize  # noqa: F401
from .model import NewsClassifier  # noqa: F401
from .dataset import FAKE, REAL, LABELS  # noqa: F401

__version__ = "1.0.0"
__all__ = ["clean_text", "tokenize", "NewsClassifier", "FAKE", "REAL", "LABELS", "__version__"]
