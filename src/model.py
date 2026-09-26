"""The fake-news classifier: TF-IDF + style features -> Logistic Regression.

Why this combination works well here:

* TF-IDF (unigrams/bigrams) captures the vocabulary and phrasing that
  separates fabricated articles from reporting.
* Style features capture the *tone* of clickbait writing (shouting caps,
  exclamation spam, "SHOCKING" style words), which generalises to unseen
  topics better than word counts alone.

The whole thing is a plain object with ``fit`` / ``predict`` / ``save`` /
``load`` so it can be used from training scripts, the Flask app and tests.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .dataset import FAKE, REAL
from .features import StyleFeatureExtractor, build_vectorizer, combine_features
from .preprocessing import PREPROCESSOR_VERSION

__all__ = ["NewsClassifier", "MODEL_VERSION"]

MODEL_VERSION = "1.0.0"


class NewsClassifier:
    """End-to-end news article classifier."""

    def __init__(self, *, max_features: int = 20_000, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = build_vectorizer(max_features=max_features)
        self.style = StyleFeatureExtractor()
        self.classifier = LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            C=4.0,
            solver="liblinear",
            random_state=random_state,
        )
        self.classes_: np.ndarray | None = None
        self.metrics_: Dict[str, float] = {}
        self.trained_at: str | None = None
        self.n_samples_: int = 0

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "NewsClassifier":
        texts = list(texts)
        labels = np.asarray(labels)

        tfidf = self.vectorizer.fit_transform(texts)
        style = self.style.fit_transform(texts)
        features = combine_features(tfidf, style)

        self.classifier.fit(features, labels)
        self.classes_ = self.classifier.classes_
        self.n_samples_ = len(texts)
        self.trained_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return self

    def fit_evaluate(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        test_texts: Sequence[str],
        test_labels: Sequence[str],
    ) -> Dict[str, float]:
        """Fit on training data and score on held-out data."""
        self.fit(train_texts, train_labels)
        self.metrics_ = self.evaluate(test_texts, test_labels)
        return self.metrics_

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _features(self, texts: Sequence[str]):
        tfidf = self.vectorizer.transform(texts)
        style = self.style.transform(list(texts))
        return combine_features(tfidf, style)

    def predict(self, texts: Sequence[str]) -> List[str]:
        self._check_fitted()
        return [str(p) for p in self.classifier.predict(self._features(texts))]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, 2)`` array ordered as :attr:`classes_`."""
        self._check_fitted()
        return self.classifier.predict_proba(self._features(texts))

    def predict_one(self, text: str) -> Dict[str, object]:
        """Classify a single article and return a JSON friendly dict."""
        proba = self.predict_proba([text])[0]
        best = int(np.argmax(proba))
        label = str(self.classes_[best])
        confidence = float(proba[best])
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_fake": label == FAKE,
            "probabilities": {
                str(cls): round(float(score), 4) for cls, score in zip(self.classes_, proba)
            },
        }

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
        self._check_fitted()
        labels = np.asarray(labels)
        predictions = np.asarray(self.predict(texts))
        proba = self.predict_proba(texts)

        scores: Dict[str, float] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        }

        # ROC-AUC needs a single continuous score; use P(FAKE).
        fake_index = int(np.where(self.classes_ == FAKE)[0][0]) if FAKE in self.classes_ else 1
        try:
            scores["roc_auc"] = float(roc_auc_score(labels == FAKE, proba[:, fake_index]))
        except ValueError:
            scores["roc_auc"] = 0.0

        matrix = confusion_matrix(labels, predictions, labels=self.classes_)
        scores["confusion_matrix"] = matrix.tolist()
        scores["report"] = classification_report(
            labels, predictions, labels=self.classes_, zero_division=0
        )
        scores["n_test"] = int(len(labels))
        return scores

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "style": self.style,
            "classifier": self.classifier,
            "classes": self.classes_,
            "metrics": self.metrics_,
            "trained_at": self.trained_at,
            "n_samples": self.n_samples_,
            "model_version": MODEL_VERSION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "max_features": self.max_features,
        }
        joblib.dump(payload, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str) -> "NewsClassifier":
        payload = joblib.load(path)
        model = cls(max_features=payload.get("max_features", 20_000))
        model.vectorizer = payload["vectorizer"]
        model.style = payload["style"]
        model.classifier = payload["classifier"]
        model.classes_ = payload.get("classes")
        model.metrics_ = payload.get("metrics", {})
        model.trained_at = payload.get("trained_at")
        model.n_samples_ = payload.get("n_samples", 0)
        return model

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("Model is not trained yet. Run `python scripts/train.py` first.")

    @property
    def is_fitted(self) -> bool:
        return self.classes_ is not None

    @property
    def feature_count(self) -> int:
        if not self.is_fitted:
            return 0
        return int(self.classifier.coef_.shape[1])

    def summary(self) -> Dict[str, object]:
        return {
            "model_version": MODEL_VERSION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "algorithm": "LogisticRegression(TF-IDF 1-2grams + style features)",
            "trained_at": self.trained_at,
            "n_training_samples": self.n_samples_,
            "n_features": self.feature_count,
            "classes": [str(c) for c in (self.classes_ if self.classes_ is not None else [])],
            "metrics": {k: v for k, v in self.metrics_.items() if k not in {"report", "confusion_matrix"}},
        }
