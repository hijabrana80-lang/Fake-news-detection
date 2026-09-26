"""Feature engineering for the fake-news classifier.

Two feature families are combined:

1. TF-IDF over unigrams/bigrams  -> lexical signal
2. Hand crafted style statistics  -> sensationalist writing cues

Style features catch things TF-IDF misses: all-caps shouting, excessive
punctuation, clickbait phrasing and unusual word-length distribution.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import clean_text

__all__ = ["StyleFeatureExtractor", "build_vectorizer", "combine_features", "STYLE_FEATURES"]

STYLE_FEATURES: List[str] = [
    "exclaim_ratio",
    "question_ratio",
    "allcaps_ratio",
    "avg_word_length",
    "type_token_ratio",
    "sensational_hits",
    "quote_ratio",
    "digit_ratio",
]

_CLICKBAIT_RE = re.compile(
    r"\b("
    r"shocking|breaking|exclusive|revealed|you won't believe|"
    r"secret|conspiracy|miracle|cure|instantly|banned|censored|"
    r"truth they|wake up|share before|must see|do not want you"
    r")\b",
    re.IGNORECASE,
)


class StyleFeatureExtractor:
    """Computes the hand-crafted style statistics as a dense array."""

    def fit(self, texts: List[str], y=None):  # noqa: ARG002 - sklearn API
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        rows = [self._one(t) for t in texts]
        return np.asarray(rows, dtype=np.float64)

    def fit_transform(self, texts: List[str], y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    @staticmethod
    def _one(raw: str) -> List[float]:
        raw = raw or ""
        cleaned = clean_text(raw)
        words: List[str] = cleaned.split()

        length = max(len(raw), 1)
        n_words = max(len(words), 1)

        exclaim = raw.count("!") / length
        question = raw.count("?") / length
        quote = raw.count('"') + raw.count("'")
        quote_ratio = quote / length

        caps_words = [w for w in raw.split() if w.isupper() and len(w) > 1]
        allcaps_ratio = len(caps_words) / n_words

        avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0
        type_token_ratio = len(set(words)) / n_words

        sensational = len(_CLICKBAIT_RE.findall(cleaned)) / n_words

        digits = sum(c.isdigit() for c in raw) / length

        return [
            exclaim,
            question,
            allcaps_ratio,
            avg_word_length,
            type_token_ratio,
            sensational,
            quote_ratio,
            digits,
        ]


class RobustTfidfVectorizer(TfidfVectorizer):
    """TF-IDF that never crashes on tiny corpora.

    scikit-learn raises ``ValueError: max_df corresponds to < documents than
    min_df`` when the corpus is smaller than the configured document-frequency
    thresholds. Small datasets are common while prototyping, so we clamp the
    thresholds for the duration of ``fit`` and restore them afterwards.
    """

    def fit(self, raw_documents, y=None):
        if isinstance(raw_documents, str):
            # Let scikit-learn raise its own, clearer error.
            return super().fit(raw_documents, y)

        documents = raw_documents if isinstance(raw_documents, (list, tuple)) else list(raw_documents)
        n_docs = len(documents)

        original_min_df, original_max_df = self.min_df, self.max_df
        try:
            min_count = original_min_df if isinstance(original_min_df, int) else original_min_df * n_docs
            max_count = original_max_df if isinstance(original_max_df, int) else original_max_df * n_docs
            if max_count < min_count:
                self.min_df = min(original_min_df, n_docs) if isinstance(original_min_df, int) else 1
                self.max_df = 1.0
            return super().fit(documents, y)
        finally:
            self.min_df, self.max_df = original_min_df, original_max_df


def build_vectorizer(*, max_features: int = 20_000) -> "RobustTfidfVectorizer":
    """TF-IDF over unigrams and bigrams with English stopwords removed."""
    return RobustTfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def combine_features(tfidf_features, style_features):
    """Horizontally stack TF-IDF and style matrices."""
    if issparse(tfidf_features):
        style = np.asarray(style_features, dtype=np.float64)
        # Scale style features so they are not drowned out by TF-IDF.
        scale = np.where(np.abs(style) > 0, np.log1p(np.abs(style)), 0.0) * np.sign(style)
        from scipy.sparse import csr_matrix

        return hstack([tfidf_features, csr_matrix(scale)])
    return np.hstack([np.asarray(tfidf_features), np.asarray(style_features, dtype=np.float64)])


def extract_style(raw_texts: List[str]) -> Dict[str, float]:
    """Convenience helper returning style features for a single document."""
    values = StyleFeatureExtractor().transform(raw_texts)
    return dict(zip(STYLE_FEATURES, values[0].tolist()))
