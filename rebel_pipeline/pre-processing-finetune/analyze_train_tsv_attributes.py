import argparse
import json
import re
from collections import Counter
from pathlib import Path

INDEXED_ATTR_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>\d+)$")

WIKI_TOKEN_VALUES = {
    "-lrb-",
    "-rrb-",
    "-lsb-",
    "-rsb-",
    "-lcb-",
    "-rcb-",
}

SPLIT_ENTITY_TOKENS = {"&", "ndash", "nbsp", "mdash", ";"}


def parse_key_value(cell: str) -> tuple[str, str] | None:
    if not cell:
        return None
    if ":" not in cell:
        return None
    key, value = cell.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def base_attribute(attr: str) -> str:
    m = INDEXED_ATTR_RE.match(attr)
    if m:
        return m.group("base")
    return attr


def analyze_tsv(path: Path) -> dict:
    raw_attr_counts = Counter()
    base_attr_counts = Counter()
    rows_with_raw_attr = Counter()
    rows_with_base_attr = Counter()
    malformed_cells = 0
    empty_lines = 0
    total_rows = 0
    total_cells = 0

    value_pattern_counts = Counter()
    schema_pattern_counts = Counter()
    attributes_with_slash = Counter()
    attributes_with_hyphen = Counter()
    column_like_attributes = Counter()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                empty_lines += 1
                continue

            total_rows += 1
            row_raw_seen = set()
            row_base_seen = set()

            for cell in line.split("\t"):
                parsed = parse_key_value(cell)
                if parsed is None:
                    malformed_cells += 1
                    continue

                attr, _ = parsed
                base_attr = base_attribute(attr)
                _, value = parsed
                total_cells += 1

                raw_attr_counts[attr] += 1
                base_attr_counts[base_attr] += 1
                row_raw_seen.add(attr)
                row_base_seen.add(base_attr)

                if attr != base_attr:
                    schema_pattern_counts["indexed_attributes"] += 1
                if "/" in base_attr:
                    schema_pattern_counts["attributes_with_slash"] += 1
                    attributes_with_slash[base_attr] += 1
                if "-" in base_attr:
                    schema_pattern_counts["attributes_with_hyphen"] += 1
                    attributes_with_hyphen[base_attr] += 1
                if base_attr.startswith("column"):
                    schema_pattern_counts["column_like_attributes"] += 1
                    column_like_attributes[base_attr] += 1

                if value == "<none>":
                    value_pattern_counts["null_marker_lt_none_gt"] += 1
                if value in WIKI_TOKEN_VALUES:
                    value_pattern_counts["wikibio_bracket_tokens"] += 1
                if value in SPLIT_ENTITY_TOKENS:
                    value_pattern_counts["split_html_entity_tokens"] += 1
                if value == "unknown":
                    value_pattern_counts["unknown_values"] += 1
                if value == "?":
                    value_pattern_counts["question_mark_values"] += 1
                if value == "or":
                    value_pattern_counts["or_token_values"] += 1
                if value == "--":
                    value_pattern_counts["double_dash_values"] += 1
                if value == ",":
                    value_pattern_counts["comma_only_values"] += 1
                if value == "/":
                    value_pattern_counts["slash_only_values"] += 1
                if value.lower() == "n/a":
                    value_pattern_counts["na_values"] += 1

            for attr in row_raw_seen:
                rows_with_raw_attr[attr] += 1
            for attr in row_base_seen:
                rows_with_base_attr[attr] += 1

    return {
        "file": str(path),
        "total_rows": total_rows,
        "total_cells": total_cells,
        "empty_lines": empty_lines,
        "malformed_cells": malformed_cells,
        "unique_raw_attributes": len(raw_attr_counts),
        "unique_base_attributes": len(base_attr_counts),
        "raw_attribute_counts": raw_attr_counts,
        "base_attribute_counts": base_attr_counts,
        "rows_with_raw_attribute": rows_with_raw_attr,
        "rows_with_base_attribute": rows_with_base_attr,
        "value_pattern_counts": value_pattern_counts,
        "schema_pattern_counts": schema_pattern_counts,
        "attributes_with_slash": attributes_with_slash,
        "attributes_with_hyphen": attributes_with_hyphen,
        "column_like_attributes": column_like_attributes,
    }


def write_outputs(stats: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "train_tsv_attribute_summary.json"
    base_list_path = output_dir / "train_tsv_base_attributes.txt"
    raw_list_path = output_dir / "train_tsv_raw_attributes.txt"

    json_summary = {
        "file": stats["file"],
        "total_rows": stats["total_rows"],
        "total_cells": stats["total_cells"],
        "empty_lines": stats["empty_lines"],
        "malformed_cells": stats["malformed_cells"],
        "unique_raw_attributes": stats["unique_raw_attributes"],
        "unique_base_attributes": stats["unique_base_attributes"],
        "schema_pattern_counts": dict(stats["schema_pattern_counts"].most_common()),
        "value_pattern_counts": dict(stats["value_pattern_counts"].most_common()),
        "column_like_attributes": dict(stats["column_like_attributes"].most_common()),
        "attributes_with_slash": dict(stats["attributes_with_slash"].most_common()),
        "attributes_with_hyphen": dict(stats["attributes_with_hyphen"].most_common()),
        "base_attribute_counts": dict(stats["base_attribute_counts"].most_common()),
        "raw_attribute_counts": dict(stats["raw_attribute_counts"].most_common()),
        "rows_with_base_attribute": dict(stats["rows_with_base_attribute"].most_common()),
        "rows_with_raw_attribute": dict(stats["rows_with_raw_attribute"].most_common()),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)

    with base_list_path.open("w", encoding="utf-8") as f:
        for attr in sorted(stats["base_attribute_counts"]):
            f.write(f"{attr}\n")

    with raw_list_path.open("w", encoding="utf-8") as f:
        for attr in sorted(stats["raw_attribute_counts"]):
            f.write(f"{attr}\n")

    return summary_path, base_list_path, raw_list_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan full train.tsv and compute attribute-name stats "
            "(raw attrs and normalized base attrs)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("wikipedia-biography-dataset/train/train.tsv"),
        help="Path to train.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated stats files",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=30,
        help="How many top base attributes to print in terminal output",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    stats = analyze_tsv(args.input)
    summary_path, base_list_path, raw_list_path = write_outputs(stats, args.output_dir)

    print(f"Input: {stats['file']}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Total key:value cells: {stats['total_cells']}")
    print(f"Unique raw attributes: {stats['unique_raw_attributes']}")
    print(f"Unique base attributes: {stats['unique_base_attributes']}")
    print(f"Malformed cells: {stats['malformed_cells']}")
    print("Schema pattern counts:")
    for key, count in stats["schema_pattern_counts"].most_common():
        print(f"  {key}: {count}")
    print("Value pattern counts:")
    for key, count in stats["value_pattern_counts"].most_common():
        print(f"  {key}: {count}")
    print("Top base attributes by total cell count:")
    for attr, count in stats["base_attribute_counts"].most_common(max(args.print_top, 0)):
        print(f"  {attr}: {count}")

    print(f"Wrote summary JSON: {summary_path}")
    print(f"Wrote base attribute list: {base_list_path}")
    print(f"Wrote raw attribute list: {raw_list_path}")


if __name__ == "__main__":
    main()
