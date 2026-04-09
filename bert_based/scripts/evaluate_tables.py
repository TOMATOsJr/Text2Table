#!/usr/bin/env python3
"""Evaluate predicted knowledge-table CSV against gold CSV.

Both files are expected in 2-column block format, with one person per block
separated by an empty line:

Title,<person>
relation,value
relation,value

[empty line]

Metrics reported:
1) Corpus chrF (character n-gram F-score)
2) Triple-level Precision / Recall / F1 with fuzzy matching
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class Triple:
    title: str
    relation: str
    value: str


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def is_blank_row(row: List[str]) -> bool:
    if not row:
        return True
    return all((cell or "").strip() == "" for cell in row)


def parse_table_blocks(path: Path) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Triple]]:
    """Parse 2-column block table into title->[(relation,value)] and triple list."""
    title_to_rows: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    triples: List[Triple] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    current_title: Optional[str] = None
    for row in tqdm(rows, total=len(rows), desc=f"parse:{path.name}", unit="row"):
        if is_blank_row(row):
            current_title = None
            continue

        if len(row) < 2:
            continue

        key = (row[0] or "").strip()
        value = (row[1] or "").strip()
        if not key:
            continue

        if key.lower() == "title":
            current_title = value
            title_to_rows.setdefault(current_title, [])
            continue

        if current_title is None:
            # Skip malformed rows outside a title block.
            continue

        title_to_rows[current_title].append((key, value))
        triples.append(Triple(title=current_title, relation=key, value=value))

    return dict(title_to_rows), triples


def char_ngrams(text: str, n: int) -> Counter:
    if not text:
        return Counter()
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def corpus_chrf(hyps: List[str], refs: List[str], n_max: int = 6, beta: float = 2.0) -> float:
    """Compute corpus chrF over aligned hypothesis-reference strings.

    Returned value is in [0, 100].
    """
    if not hyps or not refs:
        return 0.0

    num_pairs = min(len(hyps), len(refs))
    hyps = hyps[:num_pairs]
    refs = refs[:num_pairs]

    p_scores: List[float] = []
    r_scores: List[float] = []

    for n in range(1, n_max + 1):
        overlap_total = 0
        hyp_total = 0
        ref_total = 0

        for h, r in zip(hyps, refs):
            h_norm = normalize_text(h)
            r_norm = normalize_text(r)
            h_counts = char_ngrams(h_norm, n)
            r_counts = char_ngrams(r_norm, n)
            overlap = sum(min(c, r_counts[g]) for g, c in h_counts.items())

            overlap_total += overlap
            hyp_total += sum(h_counts.values())
            ref_total += sum(r_counts.values())

        p_n = overlap_total / hyp_total if hyp_total > 0 else 0.0
        r_n = overlap_total / ref_total if ref_total > 0 else 0.0
        p_scores.append(p_n)
        r_scores.append(r_n)

    p = sum(p_scores) / len(p_scores)
    r = sum(r_scores) / len(r_scores)
    if p == 0.0 and r == 0.0:
        return 0.0
    beta2 = beta * beta
    f = (1 + beta2) * p * r / (beta2 * p + r)
    return 100.0 * f


def token_f1(a: str, b: str) -> float:
    a_toks = normalize_text(a).split()
    b_toks = normalize_text(b).split()
    if not a_toks or not b_toks:
        return 0.0

    a_c = Counter(a_toks)
    b_c = Counter(b_toks)
    overlap = sum(min(a_c[t], b_c[t]) for t in a_c)
    if overlap == 0:
        return 0.0

    p = overlap / sum(a_c.values())
    r = overlap / sum(b_c.values())
    return 2 * p * r / (p + r)


def triple_similarity(pred: Triple, gold: Triple) -> float:
    # Title must match exactly after normalization.
    if normalize_text(pred.title) != normalize_text(gold.title):
        return 0.0

    rel_sim = SequenceMatcher(None, normalize_text(pred.relation), normalize_text(gold.relation)).ratio()
    val_seq = SequenceMatcher(None, normalize_text(pred.value), normalize_text(gold.value)).ratio()
    val_tok = token_f1(pred.value, gold.value)
    val_sim = max(val_seq, val_tok)

    # Put more weight on value, since relation label can vary slightly.
    return 0.35 * rel_sim + 0.65 * val_sim


def fuzzy_triple_prf(
    pred_triples: List[Triple],
    gold_triples: List[Triple],
    threshold: float,
) -> Dict[str, float]:
    """Greedy one-to-one fuzzy matching for triple P/R/F1."""
    pred_by_title: Dict[str, List[Triple]] = defaultdict(list)
    gold_by_title: Dict[str, List[Triple]] = defaultdict(list)

    for t in pred_triples:
        pred_by_title[normalize_text(t.title)].append(t)
    for t in gold_triples:
        gold_by_title[normalize_text(t.title)].append(t)

    matched = 0
    pred_total = len(pred_triples)
    gold_total = len(gold_triples)

    for title_key, preds in tqdm(pred_by_title.items(), total=len(pred_by_title), desc="match:titles", unit="title"):
        golds = gold_by_title.get(title_key, [])
        if not golds:
            continue

        used_gold = set()
        for p in preds:
            best_idx = None
            best_score = 0.0
            for gi, g in enumerate(golds):
                if gi in used_gold:
                    continue
                s = triple_similarity(p, g)
                if s > best_score:
                    best_score = s
                    best_idx = gi

            if best_idx is not None and best_score >= threshold:
                used_gold.add(best_idx)
                matched += 1

    precision = matched / pred_total if pred_total else 0.0
    recall = matched / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "matched": matched,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def title_strings_for_chrf(title_to_rows: Dict[str, List[Tuple[str, str]]]) -> Dict[str, str]:
    """Serialize each title block into deterministic text for chrF."""
    out: Dict[str, str] = {}
    for title, rows in title_to_rows.items():
        # Sort rows for deterministic comparison.
        serialized = " | ".join(f"{r}:{v}" for r, v in sorted(rows, key=lambda x: (normalize_text(x[0]), normalize_text(x[1]))))
        out[normalize_text(title)] = serialized
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction table vs gold table.")
    parser.add_argument(
        "--pred_csv",
        type=Path,
        default=Path("../tables/knowledge_graph_predictions_table.csv"),
        help="Predicted table CSV path.",
    )
    parser.add_argument(
        "--gold_csv",
        type=Path,
        default=Path("../gold/gold_test_table_filtered.csv"),
        help="Gold table CSV path.",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.78,
        help="Similarity threshold for fuzzy triple matching.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("../outputs/eval_kg_table_vs_gold_filtered.json"),
        help="Where to save evaluation results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pred_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {args.pred_csv}")
    if not args.gold_csv.exists():
        raise FileNotFoundError(f"Missing gold CSV: {args.gold_csv}")

    pred_blocks, pred_triples = parse_table_blocks(args.pred_csv)
    gold_blocks, gold_triples = parse_table_blocks(args.gold_csv)

    pred_text = title_strings_for_chrf(pred_blocks)
    gold_text = title_strings_for_chrf(gold_blocks)
    common_titles = sorted(set(pred_text).intersection(set(gold_text)))

    hyps = [pred_text[t] for t in common_titles]
    refs = [gold_text[t] for t in common_titles]
    chrf = corpus_chrf(hyps, refs)

    triple_metrics = fuzzy_triple_prf(pred_triples, gold_triples, threshold=args.fuzzy_threshold)

    result = {
        "pred_csv": str(args.pred_csv),
        "gold_csv": str(args.gold_csv),
        "num_pred_titles": len(pred_blocks),
        "num_gold_titles": len(gold_blocks),
        "num_common_titles": len(common_titles),
        "num_pred_triples": len(pred_triples),
        "num_gold_triples": len(gold_triples),
        "chrf": chrf,
        "fuzzy_threshold": args.fuzzy_threshold,
        "triple_level": triple_metrics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved evaluation: {args.output_json}")
    print(f"chrF: {chrf:.2f}")
    print(
        "Triple-level P/R/F1: "
        f"{triple_metrics['precision']:.4f} / {triple_metrics['recall']:.4f} / {triple_metrics['f1']:.4f}"
    )
    print(
        "Matched/Pred/Gold: "
        f"{triple_metrics['matched']} / {triple_metrics['pred_total']} / {triple_metrics['gold_total']}"
    )


if __name__ == "__main__":
    main()
