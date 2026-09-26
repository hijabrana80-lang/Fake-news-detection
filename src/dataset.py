"""Dataset loading and preparation.

Accepts any CSV placed in ``data/raw``:

* ``text``-style columns are auto-detected (``text``, ``title``, ``content``, ...)
* labels are auto-detected from a label column, or inferred from the file name
  (the ISOT convention ``Fake.csv`` / ``True.csv`` works out of the box)

Everything downstream only ever sees a two column frame: ``text`` and ``label``.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from .preprocessing import clean_text

__all__ = ["FAKE", "REAL", "load_raw_dataframe", "prepare_dataset", "LABELS"]

FAKE = "FAKE"
REAL = "REAL"
LABELS = [FAKE, REAL]

TEXT_COLUMNS = ("text", "title", "content", "article", "statement", "news", "body", "document")
LABEL_COLUMNS = ("label", "class", "target", "y", "fake", "verdict", "is_fake")

_TRUE_VALUES = {"real", "true", "1", "1.0", "realnews", " reliable", "reliable", "correct"}
_FAKE_VALUES = {"fake", "false", "0", "0.0", "fakenews", "unreliable", "incorrect", "satire"}


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def _normalise_label(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return REAL
    if text in _FAKE_VALUES:
        return FAKE
    return None


def _label_from_filename(path: str) -> str | None:
    name = os.path.splitext(os.path.basename(path))[0].lower()
    if re.search(r"\b(fake|false|pants|hoax)\b", name):
        return FAKE
    if re.search(r"\b(true|real|truth|verified)\b", name):
        return REAL
    return None


def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV trying a couple of encodings — news scrapes are messy."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def load_raw_dataframe(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate every labelled CSV found in *raw_dir*."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir!r}. "
            "Add a dataset (e.g. Fake.csv + True.csv) or run `python scripts/download_data.py`."
        )

    frames: List[pd.DataFrame] = []
    for path in paths:
        df = _read_csv(path)
        if df.empty:
            continue

        text_col = _find_column(df, TEXT_COLUMNS)
        if text_col is None:
            continue

        label_col = _find_column(df, LABEL_COLUMNS)
        if label_col is not None:
            labels = df[label_col].map(_normalise_label)
        else:
            fallback = _label_from_filename(path)
            labels = pd.Series([fallback] * len(df), index=df.index)

        part = pd.DataFrame({"text": df[text_col], "label": labels})
        frames.append(part)

    if not frames:
        raise ValueError(
            f"Could not find a usable text column in {raw_dir!r}. "
            f"Expected one of {TEXT_COLUMNS}."
        )

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)
    data = data[data["text"].map(clean_text).str.len() > 40]
    data = data.drop_duplicates(subset=["text"])
    data = data[data["label"].isin(LABELS)]
    return data.reset_index(drop=True)


def prepare_dataset(
    raw_dir: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Return ``(train, test, meta)`` with stratified train/test splits."""
    from sklearn.model_selection import train_test_split

    data = load_raw_dataframe(raw_dir)
    if len(data) < 50:
        raise ValueError(f"Dataset too small ({len(data)} rows) to train a model.")

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    meta = {
        "total": int(len(data)),
        "train": int(len(train)),
        "test": int(len(test)),
        **{f"train_{label.lower()}": int((train["label"] == label).sum()) for label in LABELS},
        **{f"test_{label.lower()}": int((test["label"] == label).sum()) for label in LABELS},
    }
    return train.reset_index(drop=True), test.reset_index(drop=True), meta
