import argparse
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

INDEXED_ATTR_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>\d+)$")
TEMPLATE_RE = re.compile(r"\{\{.*?\}\}")

# Pattern 1 filler tokens to skip while reconstructing values.
FILLER_TOKENS = {
    ",",
    ".",
    "-",
    "--",
    "and",
    "or",
    "the",
    "of",
    "a",
    "-lrb-",
    "-rrb-",
    "&",
    "ndash",
    "nbsp",
    "mdash",
    ";",
}

# Pattern 6 keys to drop from relation emission.
DROP_KEYS = {
    "rowspan",
    "colspan",
    "imagesize",
    "image_size",
    "headercolor",
    "header",
    "class",
    "style",
    "align",
    "width",
    "height",
    "color",
    "caption",
    "signature",
    "footnotes",
    "website",
    "source",
    "date",
    "image",
    "columns",
    "column",
    "deliveries",
    "pb",
}

SEGMENTS = [
    "international",
    "predecessor",
    "successor",
    "stumpings",
    "deliveries",
    "article",
    "service",
    "commands",
    "nickname",
    "fullname",
    "against",
    "battles",
    "wickets",
    "catches",
    "matches",
    "fivefor",
    "tenfor",
    "batting",
    "bowling",
    "country",
    "branch",
    "office",
    "height",
    "years",
    "title",
    "birth",
    "debut",
    "first",
    "place",
    "runs",
    "role",
    "club",
    "test",
    "last",
    "date",
    "year",
    "cap",
    "ft",
    "inch",
    "name",
    "shirt",
    "weight",
    "rank",
    "unit",
    "alma",
    "mater",
    "term",
    "start",
    "end",
    "party",
    "odi",
]

SEGMENTS = sorted(set(SEGMENTS), key=len, reverse=True)


def clean_markup(text: str) -> str:
    """Pattern 6 cleanup before parsing key/value tokens."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("}}|", " ")
    text = TEMPLATE_RE.sub(" ", text)
    text = text.replace("[[", "").replace("]]", "")
    text = html.unescape(text)
    # Remove wiki formatting quotes/backticks while keeping regular apostrophes in names.
    text = text.replace("``", " ").replace("''", " ").replace("`", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_leading_pipes(text: str) -> int:
    count = 0
    for ch in text:
        if ch == "|":
            count += 1
        else:
            break
    return count


def parse_key_value(cell: str) -> tuple[str, str] | None:
    if not cell or ":" not in cell:
        return None
    key, value = cell.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def split_indexed_key(key: str) -> tuple[str, int | None]:
    m = INDEXED_ATTR_RE.match(key)
    if not m:
        return key, None
    return m.group("base"), int(m.group("idx"))


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_value(value: str, drop_filler: bool = True) -> list[str]:
    tokens: list[str] = []
    for tok in normalize_spaces(value).split(" "):
        if not tok:
            continue
        t = tok.strip()
        t_low = t.lower()
        if drop_filler and t_low in FILLER_TOKENS:
            continue
        tokens.append(t)
    return tokens


def canonical_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace(" ", "_")
    key = key.replace("/", "_")
    key = key.replace("-", "_")
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def split_concatenated_component(component: str) -> list[str]:
    if not component:
        return []
    out: list[str] = []
    i = 0
    while i < len(component):
        matched = False
        for seg in SEGMENTS:
            if component.startswith(seg, i):
                out.append(seg)
                i += len(seg)
                matched = True
                break
        if not matched:
            i += 1
    if not out:
        return [component]
    return out


def normalize_relation_name(base_key: str) -> str:
    key = canonical_key(base_key)
    if not key:
        return "unknown"

    pieces: list[str] = []
    for part in key.split("_"):
        if not part:
            continue
        pieces.extend(split_concatenated_component(part))

    rel = "_".join(pieces)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel or "unknown"


def merge_indexed_tokens(entries: list[dict[str, Any]], drop_filler: bool = True) -> str:
    # Sort by suffix first, then by original row position for stable reconstruction.
    entries_sorted = sorted(
        entries,
        key=lambda e: (
            e["suffix"] if e["suffix"] is not None else 10**9,
            e["order"],
        ),
    )
    merged_tokens: list[str] = []
    for e in entries_sorted:
        merged_tokens.extend(tokenize_value(e["value"], drop_filler=drop_filler))
    return normalize_spaces(" ".join(merged_tokens))


def extract_column_labels(groups: dict[tuple[str, int, int], list[dict[str, Any]]]) -> tuple[dict[int, str], int | None]:
    labels: dict[int, str] = {}
    column_count: int | None = None

    for (base_key, col_index, _sub_id), entries in groups.items():
        if col_index != 0:
            continue

        if base_key == "columns":
            text = merge_indexed_tokens(entries, drop_filler=True)
            if text.isdigit():
                column_count = int(text)

        if base_key == "column":
            by_suffix: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for e in entries:
                if e["suffix"] is not None:
                    by_suffix[e["suffix"]].append(e)
            for idx, idx_entries in by_suffix.items():
                raw_label = merge_indexed_tokens(idx_entries, drop_filler=True)
                label = normalize_relation_name(raw_label)
                if label:
                    labels[idx] = label

    return labels, column_count


def is_column_vector_group(
    base_key: str,
    entries: list[dict[str, Any]],
    column_labels: dict[int, str],
    column_count: int | None,
) -> bool:
    if base_key in {"article_title", "name", "column", "columns"}:
        return False

    suffixes = sorted({e["suffix"] for e in entries if e["suffix"] is not None})
    if len(suffixes) < 2:
        return False

    if column_labels:
        label_keys = set(column_labels.keys())
        if set(suffixes).issubset(label_keys):
            return True

    if column_count and len(suffixes) == column_count:
        return True

    return False


def read_record_stream(
    tsv_path: Path,
    sent_path: Path,
    nb_path: Path,
    title_path: Path,
):
    with tsv_path.open("r", encoding="utf-8", errors="replace") as f_tsv, \
        sent_path.open("r", encoding="utf-8", errors="replace") as f_sent, \
        nb_path.open("r", encoding="utf-8", errors="replace") as f_nb, \
        title_path.open("r", encoding="utf-8", errors="replace") as f_title:

        sent_iter = iter(f_sent)

        for rec_id, (tsv_line, nb_line, title_line) in enumerate(
            zip(f_tsv, f_nb, f_title), start=1
        ):
            nb_text = nb_line.strip()
            if not nb_text.isdigit():
                continue
            sentence_count = int(nb_text)

            sentences: list[str] = []
            for _ in range(sentence_count):
                sent_line = next(sent_iter, "")
                sent_line = sent_line.strip()
                if sent_line:
                    sentences.append(sent_line)

            input_text = normalize_spaces(" ".join(sentences))
            yield {
                "record_id": rec_id,
                "tsv": tsv_line.rstrip("\n"),
                "title": title_line.strip(),
                "input_text": input_text,
            }


def build_triplets_from_row(
    row_tsv: str,
) -> tuple[str, list[tuple[str, str, str]], dict[str, int]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    counters = {
        "cells_total": 0,
        "cells_parsed": 0,
        "cells_skipped_none": 0,
        "cells_skipped_drop_key": 0,
        "cells_skipped_filler": 0,
    }

    subrecord_counter = 0
    previous_subrecord = False

    for order, raw_cell in enumerate(row_tsv.split("\t")):
        raw_cell = raw_cell.strip()
        if not raw_cell:
            continue
        counters["cells_total"] += 1

        cell = clean_markup(raw_cell)
        if not cell:
            continue

        pipe_count = count_leading_pipes(cell)
        stripped = cell[pipe_count:].strip()

        parsed = parse_key_value(stripped)
        if parsed is None:
            continue
        counters["cells_parsed"] += 1

        key, value = parsed
        value = clean_markup(value)
        if not value:
            counters["cells_skipped_filler"] += 1
            continue

        base_key_raw, suffix_idx = split_indexed_key(key)
        base_key = canonical_key(base_key_raw)
        if not base_key:
            continue

        col_index = pipe_count // 3
        is_subrecord = (pipe_count % 3) != 0

        if is_subrecord:
            if (not previous_subrecord) or suffix_idx in (None, 1):
                subrecord_counter += 1
            subrecord_id = subrecord_counter
            previous_subrecord = True
        else:
            subrecord_id = 0
            previous_subrecord = False

        # Subject fields are preserved even if they appear in drop families.
        is_subject_field = base_key in {"article_title", "name"}

        # Skip empty markers.
        if value.lower() == "<none>":
            counters["cells_skipped_none"] += 1
            continue

        # Skip explicit metadata keys for relation emission.
        if (base_key in DROP_KEYS or base_key.startswith("column")) and not is_subject_field:
            counters["cells_skipped_drop_key"] += 1
            continue

        if not tokenize_value(value, drop_filler=True) and not is_subject_field:
            counters["cells_skipped_filler"] += 1
            continue

        groups[(base_key, col_index, subrecord_id)].append(
            {
                "suffix": suffix_idx,
                "order": order,
                "value": value,
                "pipe_count": pipe_count,
            }
        )

    # Subject extraction: article_title first, then name.
    subject = ""
    for candidate in ("article_title", "name"):
        subject_groups = [
            entries
            for (base_key, col_index, _sub_id), entries in groups.items()
            if base_key == candidate and col_index == 0
        ]
        if subject_groups:
            # Prefer the longest merged version from available groups.
            merged = [merge_indexed_tokens(sg, drop_filler=True) for sg in subject_groups]
            merged = [m for m in merged if m]
            if merged:
                subject = max(merged, key=len)
                break

    if not subject:
        return "", [], counters

    column_labels, column_count = extract_column_labels(groups)

    triplets: list[tuple[str, str, str]] = []
    for (base_key, col_index, subrecord_id), entries in sorted(
        groups.items(), key=lambda kv: min(e["order"] for e in kv[1])
    ):
        if base_key in {"article_title", "name", "column", "columns"}:
            continue

        relation_base = normalize_relation_name(base_key)
        if not relation_base:
            continue

        use_column_mode = is_column_vector_group(
            base_key, entries, column_labels, column_count
        )

        if use_column_mode:
            by_suffix: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for e in entries:
                if e["suffix"] is not None:
                    by_suffix[e["suffix"]].append(e)

            for suffix in sorted(by_suffix):
                obj = merge_indexed_tokens(by_suffix[suffix], drop_filler=True)
                if not obj:
                    continue
                qualifier = column_labels.get(suffix, f"col{suffix}")
                relation = f"{qualifier}_{relation_base}"
                if col_index > 0:
                    relation = f"col{col_index}_{relation}"
                # Keep subrecord split semantics by not merging across subrecord_id.
                triplets.append((subject, relation, obj))
            continue

        obj = merge_indexed_tokens(entries, drop_filler=True)
        if not obj:
            continue

        relation = relation_base
        if col_index > 0:
            relation = f"col{col_index}_{relation}"

        triplets.append((subject, relation, obj))

    # Deduplicate while preserving order.
    seen = set()
    deduped: list[tuple[str, str, str]] = []
    for t in triplets:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)

    return subject, deduped, counters


def linearize_rebel(triplets: list[tuple[str, str, str]]) -> str:
    parts: list[str] = []
    for subj, rel, obj in triplets:
        s = normalize_spaces(subj)
        r = normalize_spaces(rel)
        o = normalize_spaces(obj)
        if not s or not r or not o:
            continue
        parts.append(f"<triplet> {s} <subj> {o} <obj> {r}")
    return " ".join(parts)


def process_split(
    tsv_path: Path,
    sent_path: Path,
    nb_path: Path,
    title_path: Path,
    output_path: Path,
    output_format: str,
    min_triplets: int,
    min_input_words: int,
    min_label_tokens: int,
    limit: int | None,
    dropped_output_path: Path | None,
    dropped_limit: int,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "records_total": 0,
        "records_written": 0,
        "dropped_no_subject": 0,
        "dropped_triplet_count": 0,
        "dropped_short_input": 0,
        "dropped_short_labels": 0,
        "triplets_total": 0,
    }

    rows_out: list[dict[str, str]] = []
    dropped_rows: list[dict[str, Any]] = []

    def record_drop(
        reason: str,
        row_obj: dict[str, Any],
        subject: str,
        triplets: list[tuple[str, str, str]],
        labels: str = "",
    ) -> None:
        if dropped_limit >= 0 and len(dropped_rows) >= dropped_limit:
            return
        dropped_rows.append(
            {
                "record_id": row_obj["record_id"],
                "reason": reason,
                "title": row_obj["title"],
                "subject": subject,
                "input_word_count": len(row_obj["input_text"].split()),
                "triplet_count": len(triplets),
                "label_token_count": len(labels.split()) if labels else 0,
                "input_preview": row_obj["input_text"][:300],
                "tsv_preview": row_obj["tsv"][:400],
            }
        )

    if output_format == "jsonl":
        writer = output_path.open("w", encoding="utf-8")
    else:
        writer = None

    try:
        for row in read_record_stream(tsv_path, sent_path, nb_path, title_path):
            if limit is not None and stats["records_total"] >= limit:
                break
            stats["records_total"] += 1

            input_text = row["input_text"]
            subject, triplets, _counters = build_triplets_from_row(row["tsv"])

            if not subject:
                stats["dropped_no_subject"] += 1
                record_drop("no_subject", row, subject, triplets)
                continue

            if len(triplets) < min_triplets:
                stats["dropped_triplet_count"] += 1
                record_drop("triplet_count", row, subject, triplets)
                continue

            if len(input_text.split()) < min_input_words:
                stats["dropped_short_input"] += 1
                record_drop("short_input", row, subject, triplets)
                continue

            labels = linearize_rebel(triplets)
            if len(labels.split()) < min_label_tokens:
                stats["dropped_short_labels"] += 1
                record_drop("short_labels", row, subject, triplets, labels=labels)
                continue

            out_obj = {
                "input_ids": input_text,
                "labels": labels,
            }

            stats["records_written"] += 1
            stats["triplets_total"] += len(triplets)

            if output_format == "jsonl":
                writer.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            else:
                rows_out.append(out_obj)

        if output_format == "json":
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(rows_out, f, ensure_ascii=False, indent=2)

        if dropped_output_path is not None:
            dropped_output_path.parent.mkdir(parents=True, exist_ok=True)
            with dropped_output_path.open("w", encoding="utf-8") as f:
                for item in dropped_rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        stats["dropped_report_rows"] = len(dropped_rows)
    finally:
        if writer is not None:
            writer.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WikiBio TSV + sentence files into REBEL fine-tuning triplet strings."
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("wikipedia-biography-dataset/train/train.tsv"),
        help="Path to split TSV file (structured infobox rows)",
    )
    parser.add_argument(
        "--sent",
        type=Path,
        default=Path("wikipedia-biography-dataset/train/train.sent"),
        help="Path to split sentence file",
    )
    parser.add_argument(
        "--nb",
        type=Path,
        default=Path("wikipedia-biography-dataset/train/train.nb"),
        help="Path to split nb file",
    )
    parser.add_argument(
        "--title",
        type=Path,
        default=Path("wikipedia-biography-dataset/train/train.title"),
        help="Path to split title file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pre-processing/train_rebel_ready.jsonl"),
        help="Output path",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format",
    )
    parser.add_argument(
        "--min-triplets",
        type=int,
        default=2,
        help="Skip records with fewer triplets than this",
    )
    parser.add_argument(
        "--min-input-words",
        type=int,
        default=10,
        help="Skip records with fewer input words than this",
    )
    parser.add_argument(
        "--min-label-tokens",
        type=int,
        default=5,
        help="Skip records with fewer label tokens than this",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to process",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("pre-processing/train_rebel_ready.stats.json"),
        help="Path for run statistics JSON",
    )
    parser.add_argument(
        "--dropped-output",
        type=Path,
        default=None,
        help="Optional JSONL path to write dropped records with reasons",
    )
    parser.add_argument(
        "--dropped-limit",
        type=int,
        default=200000,
        help="Max dropped records to write; set -1 to write all dropped records",
    )
    args = parser.parse_args()

    for p in (args.tsv, args.sent, args.nb, args.title):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file: {p}")

    stats = process_split(
        tsv_path=args.tsv,
        sent_path=args.sent,
        nb_path=args.nb,
        title_path=args.title,
        output_path=args.output,
        output_format=args.output_format,
        min_triplets=args.min_triplets,
        min_input_words=args.min_input_words,
        min_label_tokens=args.min_label_tokens,
        limit=args.limit,
        dropped_output_path=args.dropped_output,
        dropped_limit=args.dropped_limit,
    )

    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote dataset: {args.output}")
    print(f"Wrote stats: {args.stats_output}")
    if args.dropped_output is not None:
        print(f"Wrote dropped report: {args.dropped_output}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
