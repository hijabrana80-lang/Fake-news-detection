"""Download a public fake-news dataset into ``data/raw``.

Default source: the ISOT Fake/Real news corpus mirrored on GitHub
(Fake.csv + True.csv, ~1000 labelled articles each).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset liar
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

SOURCES = {
    "isot": [
        (
            "https://raw.githubusercontent.com/AmirhosseinHonardoust/"
            "Fake-News-Detector/main/data/Fake.csv",
            "Fake.csv",
        ),
        (
            "https://raw.githubusercontent.com/AmirhosseinHonardoust/"
            "Fake-News-Detector/main/data/True.csv",
            "True.csv",
        ),
    ],
    "liar": [
        ("https://www.cs.ucsb.edu/~william/data/liar_dataset.zip", "__zip__"),
    ],
}


def _download(url: str, destination: str) -> int:
    print(f"  -> {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with open(destination, "wb") as handle:
        handle.write(payload)
    print(f"     saved {destination} ({len(payload):,} bytes)")
    return len(payload)


def _download_zip(url: str) -> None:
    print(f"  -> {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            target = os.path.join(RAW_DIR, os.path.basename(name))
            with archive.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"     extracted {os.path.basename(name)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a fake-news dataset")
    parser.add_argument("--dataset", choices=sorted(SOURCES), default="isot")
    args = parser.parse_args(argv)

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading dataset '{args.dataset}' into {RAW_DIR}")

    for url, filename in SOURCES[args.dataset]:
        try:
            if filename == "__zip__":
                _download_zip(url)
            else:
                _download(url, os.path.join(RAW_DIR, filename))
        except Exception as exc:
            print(f"  !! failed: {exc}", file=sys.stderr)
            return 1

    print("Done. You can now run: python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
