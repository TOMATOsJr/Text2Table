#!/usr/bin/env python3
"""Build gold table blocks from cleaned_dataset/test.csv.

Output format (2 columns, no header):
Title,<title>
relation_1,<value>
relation_2,<value>

[blank line]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

NONE_VALUES = {"<none>", "none", "", "nan", "null", "n/a"}
BOX_KEY_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*):")

# Drop noisy/non-gold metadata fields.
EXCLUDED_RELATIONS = {
    "name",
    "title",
    "article_title",
    "image",
    "image_size",
    "imagesize",
    "caption",
    "portrait",
    "signature",
    "logo",
    "website",
    "url",
    "shortdescription",
    "footnotes",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_none_like(value: str) -> bool:
    return value.strip().lower() in NONE_VALUES


def parse_box_pairs(box: str) -> List[Tuple[str, str]]:
    matches = list(BOX_KEY_RE.finditer(box))
    if not matches:
        return []

    pairs: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(box)
        value = normalize_space(box[start:end])
        pairs.append((key, value))
    return pairs


def clean_join(parts: Sequence[str]) -> str:
    text = normalize_space(" ".join(p for p in parts if p))
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" ;", ";")
    text = text.replace(" :", ":")
    return normalize_space(text)


def merge_box_fields(box: str) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)

    for key, value in parse_box_pairs(box):
        base_key = re.sub(r"_\d+$", "", key)
        if base_key in EXCLUDED_RELATIONS:
            continue
        if is_none_like(value):
            continue
        grouped[base_key].append(value)

    merged: Dict[str, List[str]] = {}
    for key, parts in grouped.items():
        merged_value = clean_join(parts)
        if not merged_value or is_none_like(merged_value):
            continue
        merged[key] = [merged_value]
    return merged


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold 2-column table from test.csv box data.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("../cleaned_dataset/test.csv"),
        help="Path to test.csv with box/title columns.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("../gold/gold_test_table.csv"),
        help="Path to output 2-column CSV in gold folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input file: {args.input_csv}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(args.input_csv)

    people = 0
    written_rows = 0

    with args.input_csv.open("r", encoding="utf-8", newline="") as fin, args.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)

        for row in tqdm(reader, total=total_rows, desc="gold:test", unit="person"):
            title = normalize_space(str(row.get("title", "")))
            box = str(row.get("box", ""))
            if not title:
                continue

            merged = merge_box_fields(box)

            writer.writerow(["Title", title])
            written_rows += 1

            for relation, values in merged.items():
                for value in values:
                    writer.writerow([relation, value])
                    written_rows += 1

            writer.writerow([])
            people += 1

    print(f"Saved gold table: {args.output_csv}")
    print(f"People written: {people}")
    print(f"Data rows written (excluding blank lines): {written_rows}")


if __name__ == "__main__":
    main()
