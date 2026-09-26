"""Train the fake-news classifier and write artifacts into ``models/``.

Usage:
    python scripts/train.py
    python scripts/train.py --data-dir data/raw --test-size 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dataset import prepare_dataset  # noqa: E402
from src.model import NewsClassifier  # noqa: E402
from src.service import METRICS_FILE, MODELS_DIR, model_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the fake-news classifier")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "raw"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading dataset from {args.data_dir} ...")
    try:
        train, test, meta = prepare_dataset(
            args.data_dir, test_size=args.test_size, random_state=args.random_state
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  rows: {meta['total']} total / {meta['train']} train / {meta['test']} test")
    print(f"  train labels: FAKE={meta['train_fake']} REAL={meta['train_real']}")

    model = NewsClassifier(max_features=args.max_features, random_state=args.random_state)
    print("Training TF-IDF + style feature Logistic Regression ...")
    metrics = model.fit_evaluate(
        train["text"].tolist(),
        train["label"].tolist(),
        test["text"].tolist(),
        test["label"].tolist(),
    )

    print()
    print("Hold-out results")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  roc_auc  : {metrics['roc_auc']:.4f}")
    print(f"  features : {model.feature_count:,}")
    print()
    print(metrics["report"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = model.save(model_path())
    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(
            {
                **metrics,
                "dataset": meta,
                "model": model.summary(),
            },
            handle,
            indent=2,
        )

    print(f"Saved model    -> {path}")
    print(f"Saved metrics  -> {METRICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
