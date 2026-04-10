#!/usr/bin/env python3
"""Convert infer_rebel outputs into KG2Table-compatible JSON.

This script intentionally performs only the preprocessing that KG2Table does not:
- Parse raw REBEL generated text into structured triples
- Clean sentence text formatting
- Infer a title/person name from input text
- Emit a JSON array with the schema expected by kg_to_table.py and
    kg_to_table_csv.py

It does NOT do relation normalization, anchor filtering, disambiguation,
attribute aggregation, or table rendering. Those are handled by kg_to_table.py
or kg_to_table_csv.py.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

WIKIBIO_REPLACEMENTS = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lsb-": "[",
    "-rsb-": "]",
    "-lcb-": "{",
    "-rcb-": "}",
    "``": '"',
    "''": '"',
}


def clean_sentence(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for old, new in WIKIBIO_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+--\s+", " - ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_title_from_input(text: str) -> str:
    cleaned = clean_sentence(text)
    delimiters = [" (", " is ", " was ", " are ", " were "]
    cut = len(cleaned)
    for delimiter in delimiters:
        i = cleaned.lower().find(delimiter)
        if i != -1:
            cut = min(cut, i)
    title = cleaned[:cut].strip(" ,.-:;\"'")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def parse_rebel_output(decoded_text: str) -> list[dict[str, str]]:
    decoded_text = (
        decoded_text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("\n", " ")
        .strip()
    )

    triplets: list[dict[str, str]] = []
    subject = ""
    obj = ""
    relation = ""
    state = None

    for token in decoded_text.split():
        if token == "<triplet>":
            if subject and relation and obj:
                triplets.append(
                    {
                        "subject": subject.strip(),
                        "relation": relation.strip(),
                        "object": obj.strip(),
                    }
                )
            subject = ""
            obj = ""
            relation = ""
            state = "subject"
        elif token == "<subj>":
            state = "object"
        elif token == "<obj>":
            state = "relation"
        else:
            if state == "subject":
                subject += f" {token}"
            elif state == "object":
                obj += f" {token}"
            elif state == "relation":
                relation += f" {token}"

    if subject and relation and obj:
        triplets.append(
            {
                "subject": subject.strip(),
                "relation": relation.strip(),
                "object": obj.strip(),
            }
        )

    deduped: list[dict[str, str]] = []
    seen = set()
    for triple in triplets:
        s = re.sub(r"\s+", " ", triple["subject"]).strip()
        r = re.sub(r"\s+", " ", triple["relation"]).strip().lower()
        o = re.sub(r"\s+", " ", triple["object"]).strip()
        if not s or not r or not o:
            continue
        key = (s, r, o)
        if key not in seen:
            deduped.append({"subject": s, "relation": r, "object": o})
            seen.add(key)

    return deduped


def load_json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    # First try JSONL (one object per line).
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        if records:
            return records
    except json.JSONDecodeError:
        records = []

    # Fallback for concatenated pretty-printed JSON objects.
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    out: list[dict[str, Any]] = []
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        if isinstance(obj, dict):
            out.append(obj)
        i = j
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert infer_rebel output to KG2Table-compatible JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("predictions_test.jsonl"),
        help="Path to infer_rebel output (.jsonl, concatenated objects, or JSON array)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_for_kg_to_table.json"),
        help="Output JSON array path",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="input_ids",
        help="Field containing source text",
    )
    parser.add_argument(
        "--generated-field",
        type=str,
        default="generated_text",
        help="Field containing raw REBEL generated text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N rows (0 = all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    rows = load_json_records(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_records: list[dict[str, Any]] = []
    dropped = 0

    for idx, row in enumerate(rows, start=1):
        source_text = row.get(args.input_field, "")
        generated = row.get(args.generated_field, "")

        if not isinstance(source_text, str) or not isinstance(generated, str):
            dropped += 1
            continue

        sentence = clean_sentence(source_text)
        if not sentence:
            dropped += 1
            continue

        triples = parse_rebel_output(generated)
        title = infer_title_from_input(source_text)

        record_id = idx
        meta = row.get("inference_meta")
        if isinstance(meta, dict):
            source_row_index = meta.get("source_row_index")
            if isinstance(source_row_index, int):
                record_id = source_row_index

        entities: list[dict[str, str]] = []
        if title:
            entities.append({"text": title, "label": "PERSON"})

        output_records.append(
            {
                "record_id": record_id,
                "title": title,
                "sentence": sentence,
                "entities": entities,
                "triples": triples,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(output_records)}")
    print(f"Dropped rows: {dropped}")
    print(f"Wrote KG-compatible JSON to: {args.output}")


if __name__ == "__main__":
    main()
