"""Filter REBEL triplets that are not grounded in the source sentence.

Input format:
  JSONL or JSON array of objects like test.jsonl:
    {
      "input_ids": "source sentence text",
      "labels": "<triplet> ..."
    }

The script parses the triplets, removes those whose subject and object do not
appear in the source sentence, and writes REBEL-style JSONL again so the result
can be used directly for training.

Output format:
  One JSON object per line with only:
    - input_ids
    - labels

Rows with no grounded triplets are dropped.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm


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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("-lrb-", " ").replace("-rrb-", " ")
    text = text.replace("-lsb-", " ").replace("-rsb-", " ")
    text = text.replace("-lcb-", " ").replace("-rcb-", " ")
    text = text.replace("`", " ").replace('"', " ").replace("'", " ")
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    relation = ""
    obj = ""
    state: str | None = None

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
            relation = ""
            obj = ""
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
    seen: set[tuple[str, str, str]] = set()
    for triple in triplets:
        subj = re.sub(r"\s+", " ", triple["subject"]).strip()
        rel = re.sub(r"\s+", " ", triple["relation"]).strip().lower()
        obj = re.sub(r"\s+", " ", triple["object"]).strip()
        if not subj or not rel or not obj:
            continue
        key = (subj, rel, obj)
        if key not in seen:
            deduped.append({"subject": subj, "relation": rel, "object": obj})
            seen.add(key)

    return deduped


def load_json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
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

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    index = 0
    out: list[dict[str, Any]] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, next_index = decoder.raw_decode(text, index)
        if isinstance(obj, dict):
            out.append(obj)
        index = next_index

    return out


def triple_is_grounded(triple: dict[str, str], sentence: str) -> bool:
    sent_lower = normalize_text(sentence)
    subject = normalize_text(triple["subject"])
    obj = normalize_text(triple["object"])
    return bool(subject and subject in sent_lower) or bool(obj and obj in sent_lower)


def encode_rebel_triples(triples: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for triple in triples:
        parts.append(
            f"<triplet> {triple['subject']} <subj> {triple['object']} <obj> {triple['relation']}"
        )
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter REBEL triples grounded in the source sentence and write JSONL."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path("../outputs/test.jsonl"),
        help="Path to REBEL output JSONL/JSON file.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("../outputs/knowledge_graph_predictions_grounded.jsonl"),
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="input_ids",
        help="Field containing the source sentence.",
    )
    parser.add_argument(
        "--triple-field",
        type=str,
        default="labels",
        help="Field containing the REBEL triplet string.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_json_records(path)
    if not rows:
        raise ValueError(f"No JSON records found in {path}")
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_jsonl: Path, args: argparse.Namespace) -> dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    grounded_count = 0
    dropped_count = 0

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, total=len(rows), desc="filtering", unit="row"):
            sentence = row.get(args.input_field, "")
            if not isinstance(sentence, str):
                sentence = str(sentence)

            triple_source = row.get("generated_text") or row.get(args.triple_field, "")
            if not isinstance(triple_source, str):
                triple_source = str(triple_source)

            triples = parse_rebel_output(triple_source)
            grounded_triples = [t for t in triples if triple_is_grounded(t, sentence)]

            dropped_count += len(triples) - len(grounded_triples)
            grounded_count += len(grounded_triples)
            row_count += 1

            if not grounded_triples:
                continue

            output_row = {
                "input_ids": sentence,
                "labels": encode_rebel_triples(grounded_triples),
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    return {
        "rows": row_count,
        "grounded_triples": grounded_count,
        "dropped_triples": dropped_count,
    }


def main() -> None:
    args = parse_args()

    if not args.input_json.exists():
        raise FileNotFoundError(f"Missing input file: {args.input_json}")

    rows = load_rows(args.input_json)
    stats = write_jsonl(rows, args.output_jsonl, args)

    print(f"Saved grounded JSONL: {args.output_jsonl}")
    print(f"Rows processed: {stats['rows']}")
    print(f"Grounded triples kept: {stats['grounded_triples']}")
    print(f"Dropped triples: {stats['dropped_triples']}")


if __name__ == "__main__":
    main()