#!/usr/bin/env python3
"""Canonicalize predicted relation strings using Sentence-BERT embeddings.

Input JSONL should contain a relation key (default: predicted_relation).
Output is a JSON mapping of raw relation -> canonical relation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=Path("outputs/relation_canonical_map.json"))
    parser.add_argument("--relation_key", type=str, default="predicted_relation")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--distance_threshold", type=float, default=0.25)
    parser.add_argument("--min_cluster_size", type=int, default=1)
    return parser.parse_args()


def most_frequent(items: List[str]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for it in items:
        counts[it] += 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with args.input_jsonl.open("r", encoding="utf-8") as f_count:
        total_rows = sum(1 for _ in f_count)

    relations: List[str] = []
    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_rows, desc="canonicalize:rows", unit="row"):
            row = json.loads(line)
            rel = row.get(args.relation_key)
            if rel:
                relations.append(rel)

    unique_relations = sorted(set(relations))
    if not unique_relations:
        raise ValueError("No relations found in input.")

    model = SentenceTransformer(args.model_name)
    emb = model.encode(unique_relations, normalize_embeddings=True)

    if len(unique_relations) == 1:
        mapping = {unique_relations[0]: unique_relations[0]}
    else:
        cluster = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=args.distance_threshold,
        )
        labels = cluster.fit_predict(emb)

        groups: Dict[int, List[str]] = defaultdict(list)
        for rel, c in zip(unique_relations, labels):
            groups[int(c)].append(rel)

        mapping: Dict[str, str] = {}
        for _, members in groups.items():
            if len(members) < args.min_cluster_size:
                for m in members:
                    mapping[m] = m
                continue
            canon = most_frequent(members)
            for m in members:
                mapping[m] = canon

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Unique relations: {len(unique_relations)}")
    print(f"Wrote canonical map: {args.output_json}")


if __name__ == "__main__":
    main()