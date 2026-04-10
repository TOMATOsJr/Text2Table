#!/usr/bin/env python3
"""Build per-title knowledge graphs from prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm


def build_knowledge_graph(triples: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, List[str]]]:
    """Build a simple knowledge graph from (subject, relation, object) triples."""
    kg = defaultdict(lambda: defaultdict(list))

    for subj, rel, obj in triples:
        subj = subj.strip()
        rel = rel.strip()
        obj = obj.strip()

        if not subj or not rel or not obj:
            continue

        if obj not in kg[subj][rel]:
            kg[subj][rel].append(obj)

    return {s: dict(r) for s, r in kg.items()}


def print_knowledge_graph(kg: Dict[str, Dict[str, List[str]]]) -> None:
    for subject, relations in kg.items():
        print(f"\n{subject}")
        for relation, objects in relations.items():
            for obj in objects:
                print(f"  ├── {relation} -> {obj}")


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip().lower())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown_title"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_prediction_triples(jsonl_path: Path, relation_key: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    total_rows = count_lines(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_rows, desc=f"read:{jsonl_path.name}", unit="row"):
            row = json.loads(line)
            title = str(row.get("title", "")).strip()
            relation = str(row.get(relation_key, "") or row.get("predicted_relation", "")).strip()
            value = str(row.get("span_text", "")).strip()
            if title and relation and value:
                triples.append((title, relation, value))
    return triples


def write_per_person_graphs(
    kg: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for title, relations in tqdm(kg.items(), total=len(kg), desc=f"write:{output_dir.name}", unit="person"):
        out_path = output_dir / f"{sanitize_filename(title)}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"title": title, "relations": relations}, f, indent=2, ensure_ascii=False)


def build_and_save_graphs(
    input_jsonl: Path,
    relation_key: str,
    aggregate_out: Path,
    per_person_out_dir: Path,
) -> Dict[str, Dict[str, List[str]]]:
    triples = read_prediction_triples(input_jsonl, relation_key=relation_key)
    print(f"Building graph from {len(triples)} triples ({input_jsonl.name})")
    kg = build_knowledge_graph(triples)

    aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_out.open("w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    write_per_person_graphs(kg, per_person_out_dir)
    return kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-title knowledge graphs from predictions.")
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=Path("../outputs/predictions.jsonl"),
        help="Path to predictions JSONL.",
    )
    parser.add_argument(
        "--canonical_predictions_file",
        type=Path,
        default=Path("../outputs/predictions_canonical.jsonl"),
        help="Path to canonical predictions JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../outputs"),
        help="Output directory for saved knowledge graphs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions_file.exists():
        raise FileNotFoundError(f"Missing input file: {args.predictions_file}")
    if not args.canonical_predictions_file.exists():
        raise FileNotFoundError(f"Missing input file: {args.canonical_predictions_file}")

    kg_pred = build_and_save_graphs(
        input_jsonl=args.predictions_file,
        relation_key="predicted_relation",
        aggregate_out=args.output_dir / "knowledge_graph_predictions.json",
        per_person_out_dir=args.output_dir / "knowledge_graphs" / "predictions",
    )

    kg_canon = build_and_save_graphs(
        input_jsonl=args.canonical_predictions_file,
        relation_key="canonical_relation",
        aggregate_out=args.output_dir / "knowledge_graph_predictions_canonical.json",
        per_person_out_dir=args.output_dir / "knowledge_graphs" / "predictions_canonical",
    )

    print(f"Saved aggregate graph: {args.output_dir / 'knowledge_graph_predictions.json'}")
    print(f"Saved aggregate graph: {args.output_dir / 'knowledge_graph_predictions_canonical.json'}")
    print(f"Saved per-person graphs: {args.output_dir / 'knowledge_graphs' / 'predictions'}")
    print(f"Saved per-person graphs: {args.output_dir / 'knowledge_graphs' / 'predictions_canonical'}")
    print(f"Titles in predictions graph: {len(kg_pred)}")
    print(f"Titles in canonical graph: {len(kg_canon)}")


if __name__ == "__main__":
    main()