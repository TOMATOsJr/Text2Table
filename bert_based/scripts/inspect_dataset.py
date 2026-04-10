#!/usr/bin/env python3
"""Quick inspection utility for ../../../../../cleaned_dataset CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from tqdm import tqdm


def inspect_file(path: Path, samples: int) -> None:
    print(f"\n=== {path.name} ===")
    with path.open("r", encoding="utf-8", newline="") as f_count:
        total_rows = max(sum(1 for _ in f_count) - 1, 0)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        print("columns:", reader.fieldnames)
        counts = Counter()
        sample_rows = []
        for i, row in enumerate(tqdm(reader, total=total_rows, desc=f"inspect:{path.name}", unit="row")):
            counts["rows"] += 1
            if i < samples:
                sample_rows.append(row)

    print("rows:", counts["rows"])
    for i, r in enumerate(sample_rows):
        print(f"sample[{i}] title:", str(r.get("title", ""))[:120])
        print("box:", str(r.get("box", ""))[:220])
        print("sent:", str(r.get("sent", ""))[:220])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("../../../../../cleaned_dataset"))
    parser.add_argument("--samples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in ("train.csv", "valid.csv", "test.csv"):
        fp = args.input_dir / split
        if not fp.exists():
            raise FileNotFoundError(fp)
        inspect_file(fp, args.samples)


if __name__ == "__main__":
    main()
