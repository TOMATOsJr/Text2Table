#!/usr/bin/env python3
"""Prepare WikiBio-style CSV into span extraction and relation classification datasets.

Outputs per split:
- triples_<split>.jsonl: aligned (title, relation, value, span_text, start, end, sentence)
- relation_cls_<split>.jsonl: one instance per aligned span with relation label
- span_ner_<split>.jsonl: token-level BIO labels for span extraction training
- span_spans_<split>.jsonl: sentence with all aligned attribute spans
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

NONE_VALUES = {"<none>", "none", "", "nan", "null", "n/a"}
BOX_KEY_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*):")
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
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
}


@dataclass
class AlignedValue:
    relation: str
    raw_value: str
    span_text: str
    start: int
    end: int


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_none_like(value: str) -> bool:
    return value.strip().lower() in NONE_VALUES


def parse_box_pairs(box: str) -> List[Tuple[str, str]]:
    """Parse key:value pairs from noisy WikiBio box strings.

    This parser is robust to missing separators such as:
    birth_date_1:14birth_date_2:april
    """
    matches = list(BOX_KEY_RE.finditer(box))
    if not matches:
        return []

    pairs: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
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
    text = text.replace(" ( ", " (")
    text = text.replace(" )", ")")
    return normalize_space(text)


def split_multi_values(value: str) -> List[str]:
    # Split common list separators while keeping meaningful chunks.
    chunks = re.split(r"\s*/\s*|\s*\|\s*|\s*;\s*", value)
    out = [normalize_space(c) for c in chunks if normalize_space(c)]
    return out if out else [normalize_space(value)]


def merge_box_fields(box: str, split_multi: bool = True) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)

    for key, value in parse_box_pairs(box):
        base_key = re.sub(r"_\d+$", "", key)
        if is_none_like(value):
            continue
        grouped[base_key].append(value)

    merged: Dict[str, List[str]] = {}
    for key, parts in grouped.items():
        merged_value = clean_join(parts)
        if not merged_value or is_none_like(merged_value):
            continue
        values = split_multi_values(merged_value) if split_multi else [merged_value]
        values = [v for v in values if not is_none_like(v)]
        if values:
            merged[key] = values
    return merged


def _normalized_with_map(text: str) -> Tuple[str, List[int]]:
    norm_chars: List[str] = []
    idx_map: List[int] = []
    prev_space = True

    for i, ch in enumerate(text):
        if ch.isalnum():
            # Some Unicode chars expand on lower/casefold (for example, İ -> i + combining dot).
            # Expand safely while preserving a character-to-source index map.
            folded = ch.casefold()
            emitted = False
            for out_ch in folded:
                if out_ch.isalnum():
                    norm_chars.append(out_ch)
                    idx_map.append(i)
                    emitted = True
            if emitted:
                prev_space = False
        else:
            if not prev_space:
                norm_chars.append(" ")
                idx_map.append(i)
            prev_space = True

    normalized = "".join(norm_chars).strip()
    # Rebuild map for stripped string.
    if not normalized:
        return "", []
    left_trim = 0
    while left_trim < len(norm_chars) and norm_chars[left_trim] == " ":
        left_trim += 1
    right_trim = len(norm_chars)
    while right_trim > 0 and norm_chars[right_trim - 1] == " ":
        right_trim -= 1
    return normalized, idx_map[left_trim:right_trim]


def exact_normalized_span(sentence: str, value: str) -> Optional[Tuple[int, int]]:
    sent_norm, sent_map = _normalized_with_map(sentence)
    val_norm, _ = _normalized_with_map(value)

    if not sent_norm or not val_norm:
        return None

    pos = sent_norm.find(val_norm)
    if pos == -1:
        return None

    end_idx = pos + len(val_norm) - 1
    if pos >= len(sent_map) or end_idx >= len(sent_map):
        return None

    start = sent_map[pos]
    end = sent_map[end_idx] + 1
    return start, end


def token_window_span(sentence: str, value: str, threshold: float = 0.6) -> Optional[Tuple[int, int]]:
    sent_tokens = [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(sentence)]
    val_tokens = [t.lower() for t in TOKEN_RE.findall(value)]

    if not sent_tokens or not val_tokens:
        return None

    val_counter = Counter(val_tokens)
    target_len = len(val_tokens)
    best_score = 0.0
    best_overlap = 0
    best_span: Optional[Tuple[int, int]] = None

    min_w = max(1, target_len - 2)
    max_w = min(len(sent_tokens), target_len + 2)

    for w in range(min_w, max_w + 1):
        for i in range(0, len(sent_tokens) - w + 1):
            window = sent_tokens[i : i + w]
            window_counter = Counter(tok for tok, _, _ in window)
            overlap = sum(min(val_counter[k], window_counter[k]) for k in val_counter)
            precision = overlap / max(1, sum(window_counter.values()))
            recall = overlap / max(1, sum(val_counter.values()))
            if precision + recall == 0:
                score = 0.0
            else:
                score = 2 * precision * recall / (precision + recall)

            if score > best_score:
                best_score = score
                best_overlap = overlap
                best_span = (window[0][1], window[-1][2])

    min_overlap = 1 if target_len == 1 else 2
    if best_span and best_score >= threshold and best_overlap >= min_overlap:
        return best_span
    return None


def align_value_to_sentence(sentence: str, value: str) -> Optional[Tuple[int, int, str]]:
    span = exact_normalized_span(sentence, value)
    if span is None:
        span = token_window_span(sentence, value)
    if span is None:
        return None
    start, end = span
    return start, end, sentence[start:end]


def build_bio_tags(sentence: str, spans: Sequence[Tuple[int, int]]) -> Tuple[List[str], List[str]]:
    token_matches = list(TOKEN_RE.finditer(sentence))
    tokens = [m.group(0) for m in token_matches]
    tags = ["O"] * len(tokens)

    for start, end in spans:
        covered = [
            i
            for i, m in enumerate(token_matches)
            if m.end() > start and m.start() < end
        ]
        if not covered:
            continue
        tags[covered[0]] = "B-ATTR"
        for i in covered[1:]:
            tags[i] = "I-ATTR"

    return tokens, tags


def process_split(input_csv: Path, output_dir: Path, split_name: str) -> Dict[str, int]:
    triples_path = output_dir / f"triples_{split_name}.jsonl"
    rel_path = output_dir / f"relation_cls_{split_name}.jsonl"
    span_spans_path = output_dir / f"span_spans_{split_name}.jsonl"
    span_ner_path = output_dir / f"span_ner_{split_name}.jsonl"

    stats = Counter()

    with input_csv.open("r", encoding="utf-8", newline="") as f_count:
        total_rows = max(sum(1 for _ in f_count) - 1, 0)

    with input_csv.open("r", encoding="utf-8", newline="") as fin, \
        triples_path.open("w", encoding="utf-8") as f_triples, \
        rel_path.open("w", encoding="utf-8") as f_rel, \
        span_spans_path.open("w", encoding="utf-8") as f_spans, \
        span_ner_path.open("w", encoding="utf-8") as f_ner:

        reader = csv.DictReader(fin)
        for row in tqdm(reader, total=total_rows, desc=f"{split_name}:rows", unit="row"):
            stats["rows"] += 1
            title = normalize_space(row.get("title", ""))
            sentence = row.get("sent", "")
            box = row.get("box", "")

            merged = merge_box_fields(box, split_multi=True)
            aligned_items: List[AlignedValue] = []

            for relation, values in merged.items():
                if relation in EXCLUDED_RELATIONS:
                    # Keep supervision focused on attribute extraction.
                    continue
                for value in values:
                    stats["candidate_values"] += 1
                    aligned = align_value_to_sentence(sentence, value)
                    if aligned is None:
                        stats["unaligned_values"] += 1
                        continue

                    start, end, span_text = aligned
                    item = AlignedValue(
                        relation=relation,
                        raw_value=value,
                        span_text=span_text,
                        start=start,
                        end=end,
                    )
                    aligned_items.append(item)
                    stats["aligned_values"] += 1

                    triple = {
                        "title": title,
                        "relation": relation,
                        "value": value,
                        "span_text": span_text,
                        "start": start,
                        "end": end,
                        "sentence": sentence,
                        "split": split_name,
                    }
                    f_triples.write(json.dumps(triple, ensure_ascii=False) + "\n")

                    rel_ex = {
                        "title": title,
                        "span_text": span_text,
                        "sentence": sentence,
                        "relation": relation,
                        "split": split_name,
                    }
                    f_rel.write(json.dumps(rel_ex, ensure_ascii=False) + "\n")

            span_records = [
                {
                    "relation": x.relation,
                    "value": x.raw_value,
                    "span_text": x.span_text,
                    "start": x.start,
                    "end": x.end,
                }
                for x in aligned_items
            ]

            spans_for_bio = [(x.start, x.end) for x in aligned_items]
            tokens, ner_tags = build_bio_tags(sentence, spans_for_bio)

            f_spans.write(
                json.dumps(
                    {
                        "title": title,
                        "sentence": sentence,
                        "spans": span_records,
                        "split": split_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            f_ner.write(
                json.dumps(
                    {
                        "title": title,
                        "sentence": sentence,
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                        "split": split_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if aligned_items:
                stats["rows_with_alignment"] += 1

    return dict(stats)


def write_stats(output_dir: Path, all_stats: Dict[str, Dict[str, int]]) -> None:
    with (output_dir / "preprocess_stats.json").open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("../../../../../cleaned_dataset"),
        help="Directory containing train.csv/valid.csv/test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for JSONL artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_to_file = {
        "train": args.input_dir / "train.csv",
        "valid": args.input_dir / "valid.csv",
        "test": args.input_dir / "test.csv",
    }

    all_stats: Dict[str, Dict[str, int]] = {}
    for split, file_path in split_to_file.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        print(f"Processing {split}: {file_path}")
        stats = process_split(file_path, args.output_dir, split)
        all_stats[split] = stats
        print(f"{split} stats: {stats}")

    write_stats(args.output_dir, all_stats)
    print(f"Wrote artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
