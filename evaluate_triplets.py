#!/usr/bin/env python
"""Evaluate REBEL triplet extraction against ground truth.

Compares generated triplets (subject, relation, object) against ground truth
triplets using token-overlap partial matching.

Usage:
    python evaluate_triplets.py \
        --pred outputs/triples_rebel_direct.finetuned.multigpu.sample20.jsonl \
        --gold pre-processing-finetune/rebel_dataset/test.jsonl \
        --sample 0
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_ground_truth_triplets(labels_str: str) -> list[dict[str, str]]:
    """Parse REBEL format triplets from ground truth labels.

    Format: <triplet> subject <subj> value <obj> relation
    Returns list of {"subject": ..., "object": ..., "relation": ...}
    """
    triplets = []
    parts = labels_str.split("<triplet>")

    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue

        subj_match = re.search(r"^(.+?)\s+<subj>\s+(.+?)\s+<obj>\s+(.+?)$", part)
        if subj_match:
            subject = subj_match.group(1).strip()
            obj_value = subj_match.group(2).strip()
            relation = subj_match.group(3).strip()

            subject = normalize_text(subject)
            obj_value = normalize_text(obj_value)
            relation = normalize_text(relation)

            if subject and obj_value and relation:
                triplets.append({
                    "subject": subject,
                    "object": obj_value,
                    "relation": relation,
                })

    return triplets


def load_ground_truth(gt_path: Path) -> list[list[dict[str, str]]]:
    """Load ground truth triplets from JSONL file."""
    records = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            labels = entry.get("labels", "")
            triplets = parse_ground_truth_triplets(labels)
            records.append(triplets)

    return records


def load_predictions(pred_path: Path) -> list[dict[str, list[dict[str, str]]]]:
    """Load predicted triplets from JSONL file."""
    records = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line due to JSON error: {e}")
                continue

            raw_triples = []
            for t in entry.get("raw_triples", []):
                raw_triples.append({
                    "subject": normalize_text(t.get("subject", "")),
                    "object": normalize_text(t.get("object", "")),
                    "relation": normalize_text(t.get("relation", "")),
                })

            filtered_triples = []
            for t in entry.get("filtered_triples", []):
                filtered_triples.append({
                    "subject": normalize_text(t.get("subject", "")),
                    "object": normalize_text(t.get("object", "")),
                    "relation": normalize_text(t.get("relation", "")),
                })

            records.append({
                "raw_triples": raw_triples,
                "filtered_triples": filtered_triples,
            })

    return records


def token_overlap(pred_text: str, gold_text: str, threshold: float = 0.5) -> bool:
    """Check if token overlap between pred and gold reaches threshold."""
    pred_tokens = set(pred_text.split())
    gold_tokens = set(gold_text.split())

    if not pred_tokens or not gold_tokens:
        return False

    overlap = pred_tokens & gold_tokens
    overlap_ratio = len(overlap) / max(len(pred_tokens), len(gold_tokens))

    return overlap_ratio >= threshold


def find_matching_triplet(
    pred_triplet: dict[str, str],
    gold_triplets: list[dict[str, str]],
    overlap_threshold: float = 0.5,
) -> bool:
    """Check if pred triplet matches any gold triplet using token overlap."""
    for gold_triplet in gold_triplets:
        subj_match = token_overlap(
            pred_triplet["subject"],
            gold_triplet["subject"],
            overlap_threshold
        )
        obj_match = token_overlap(
            pred_triplet["object"],
            gold_triplet["object"],
            overlap_threshold
        )
        rel_match = token_overlap(
            pred_triplet["relation"],
            gold_triplet["relation"],
            overlap_threshold
        )

        if subj_match and obj_match and rel_match:
            return True

    return False


def evaluate_triplets(
    predictions: list[dict[str, list[dict[str, str]]]],
    ground_truth: list[list[dict[str, str]]],
    sample: int = 0,
    overlap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate predictions against ground truth."""
    n_records = min(len(ground_truth), sample) if sample > 0 else len(ground_truth)

    metrics = {
        "raw_triples": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact_match_records": 0,
            "partial_match_records": 0,
        },
        "filtered_triples": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact_match_records": 0,
            "partial_match_records": 0,
        },
    }

    for idx in range(n_records):
        gold_triplets = ground_truth[idx]
        pred_data = predictions[idx] if idx < len(predictions) else {"raw_triples": [], "filtered_triples": []}

        raw_pred = pred_data.get("raw_triples", [])
        raw_tp = sum(
            1 for p in raw_pred
            if find_matching_triplet(p, gold_triplets, overlap_threshold)
        )
        raw_fp = len(raw_pred) - raw_tp
        raw_fn = len(gold_triplets) - raw_tp

        metrics["raw_triples"]["tp"] += raw_tp
        metrics["raw_triples"]["fp"] += max(0, raw_fp)
        metrics["raw_triples"]["fn"] += max(0, raw_fn)

        if len(gold_triplets) > 0 and len(raw_pred) == len(gold_triplets):
            metrics["raw_triples"]["exact_match_records"] += 1
        if raw_tp > 0:
            metrics["raw_triples"]["partial_match_records"] += 1

        filtered_pred = pred_data.get("filtered_triples", [])
        filtered_tp = sum(
            1 for p in filtered_pred
            if find_matching_triplet(p, gold_triplets, overlap_threshold)
        )
        filtered_fp = len(filtered_pred) - filtered_tp
        filtered_fn = len(gold_triplets) - filtered_tp

        metrics["filtered_triples"]["tp"] += filtered_tp
        metrics["filtered_triples"]["fp"] += max(0, filtered_fp)
        metrics["filtered_triples"]["fn"] += max(0, filtered_fn)

        if len(gold_triplets) > 0 and len(filtered_pred) == len(gold_triplets):
            metrics["filtered_triples"]["exact_match_records"] += 1
        if filtered_tp > 0:
            metrics["filtered_triples"]["partial_match_records"] += 1

    for triple_type in ["raw_triples", "filtered_triples"]:
        m = metrics[triple_type]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        m["precision"] = round(precision, 4)
        m["recall"] = round(recall, 4)
        m["f1"] = round(f1, 4)
        m["accuracy"] = round(accuracy, 4)

    return {
        "num_records": n_records,
        "overlap_threshold": overlap_threshold,
        "metrics": metrics,
    }


def print_results(results: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    n = results["num_records"]
    threshold = results["overlap_threshold"]

    print(f"\n{'='*70}")
    print(f" Triplet Evaluation Results")
    print(f" Records evaluated: {n}")
    print(f" Token overlap threshold: {threshold:.1%}")
    print(f"{'='*70}")

    for triple_type in ["raw_triples", "filtered_triples"]:
        m = results["metrics"][triple_type]
        label = triple_type.replace("_", " ").title()

        print(f"\n  {label}")
        print(f"  {'-'*70}")
        print(f"    TP: {m['tp']:>6d}   FP: {m['fp']:>6d}   FN: {m['fn']:>6d}")
        print(f"    Precision:      {m['precision']:.4f}")
        print(f"    Recall:         {m['recall']:.4f}")
        print(f"    F1 Score:       {m['f1']:.4f}")
        print(f"    Accuracy:       {m['accuracy']:.4f}")
        print(f"\n    Exact match records:   {m['exact_match_records']}/{n}")
        print(f"    Partial match records: {m['partial_match_records']}/{n}")

    print(f"\n{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate REBEL triplet extraction against ground truth."
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Path to predicted triplets JSONL file",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to ground truth JSONL file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Evaluate only first N records (0 = all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Token overlap threshold for matching (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file (optional)",
    )

    args = parser.parse_args()

    if not args.pred.exists():
        raise FileNotFoundError(f"Prediction file not found: {args.pred}")
    if not args.gold.exists():
        raise FileNotFoundError(f"Ground truth file not found: {args.gold}")

    print(f"Loading ground truth from: {args.gold}")
    ground_truth = load_ground_truth(args.gold)
    print(f"  Loaded {len(ground_truth)} records")

    print(f"Loading predictions from: {args.pred}")
    predictions = load_predictions(args.pred)
    print(f"  Loaded {len(predictions)} records")

    print(f"Evaluating...")
    results = evaluate_triplets(
        predictions,
        ground_truth,
        sample=args.sample,
        overlap_threshold=args.threshold,
    )

    print_results(results)

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            save_results = {
                "config": {
                    "pred_file": str(args.pred),
                    "gold_file": str(args.gold),
                    "sample": args.sample,
                    "threshold": args.threshold,
                },
                "results": results,
            }
            json.dump(save_results, f, indent=2)
        print(f"Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
