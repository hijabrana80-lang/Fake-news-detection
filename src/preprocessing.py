"""Text cleaning and NLP preprocessing utilities.

The pipeline is deliberately lightweight and dependency friendly:
* lowercase + unicode normalisation
* URL / email / HTML tag stripping
* number and punctuation normalisation
* optional stopword removal and lemmatisation when NLTK data is available

Everything works offline. If the NLTK corpora are missing we degrade
gracefully instead of raising, so the web app never crashes on a fresh
machine.
"""

from __future__ import annotations

import html
import re
import string
from functools import lru_cache
from typing import Iterable, List

__all__ = [
    "clean_text",
    "tokenize",
    "PREPROCESSOR_VERSION",
    "nltk_available",
]

PREPROCESSOR_VERSION = "1.0.0"

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_REDDIT_RE = re.compile(r"/r/\w+|/u/\w+")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

_PUNCT_TABLE = str.maketrans(" ", " ", string.punctuation)

_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of to in on at by for with from as is are was
    were be been being it its this that these those he she they we you i me him
    her them us our your his their my not no nor so too very can will just don
    should now
    """.split()
)


def nltk_available() -> bool:
    """Return True when the optional NLTK stopword corpus is importable."""
    try:
        from nltk.corpus import stopwords  # noqa: F401

        stopwords.words("english")
    except Exception:  # pragma: no cover - depends on local NLTK data
        return False
    return True


@lru_cache(maxsize=1)
def _stopword_set() -> frozenset[str]:
    """Load NLTK stopwords when present, otherwise use the bundled list."""
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except Exception:
        return _STOPWORDS


def clean_text(text: str | None) -> str:
    """Normalise raw news text into a machine-learning friendly string."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = _SCRIPT_RE.sub(" ", text)
    text = _URL_RE.sub(" url ", text)
    text = _EMAIL_RE.sub(" email ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _REDDIT_RE.sub(" ", text)
    text = text.lower()

    # Keep sentence-ending punctuation meaningful, drop the rest.
    text = text.replace(".", " ")
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(
    text: str | None,
    *,
    remove_stopwords: bool = True,
    min_length: int = 2,
) -> List[str]:
    """Split cleaned text into tokens, optionally dropping stopwords."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens: Iterable[str] = cleaned.split()
    if remove_stopwords:
        stops = _stopword_set()
        tokens = (t for t in tokens if t not in stops)

    return [t for t in tokens if len(t) >= min_length]
