#!/usr/bin/env python3
"""Run inference pipeline: span extractor -> relation classifier -> canonicalizer.

This script expects trained model folders from train_span_extractor.py and
train_relation_luke.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span_model_dir", type=Path, required=True)
    parser.add_argument("--relation_model_dir", type=Path, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True, help="JSONL rows with title and sentence")
    parser.add_argument("--output_jsonl", type=Path, default=Path("outputs/predictions.jsonl"))
    parser.add_argument("--canonical_map", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def predict_spans(sentence: str, tokenizer, model, max_length: int = 256) -> List[Tuple[int, int, str]]:
    token_matches = list(TOKEN_RE.finditer(sentence))
    words = [m.group(0) for m in token_matches]
    if not words:
        return []

    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred = logits.argmax(dim=-1).tolist()

    word_ids = enc.word_ids(batch_index=0)
    word_labels: Dict[int, int] = {}
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in word_labels:
            continue
        word_labels[wid] = pred[idx]

    spans: List[Tuple[int, int, str]] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    for wi in range(len(words)):
        label_id = word_labels.get(wi, 0)
        label = model.config.id2label.get(label_id, "O")
        m = token_matches[wi]

        if label == "B-ATTR":
            if current_start is not None:
                spans.append((current_start, current_end, sentence[current_start:current_end]))
            current_start, current_end = m.start(), m.end()
        elif label == "I-ATTR" and current_start is not None:
            current_end = m.end()
        else:
            if current_start is not None:
                spans.append((current_start, current_end, sentence[current_start:current_end]))
                current_start, current_end = None, None

    if current_start is not None:
        spans.append((current_start, current_end, sentence[current_start:current_end]))

    # De-duplicate exact repeated spans.
    dedup = []
    seen = set()
    for s in spans:
        key = (s[0], s[1], s[2].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    return dedup


def predict_relation(title: str, span_text: str, sentence: str, tokenizer, model, max_length: int = 256) -> str:
    text = f"title: {title} [SEP] span: {span_text} [SEP] sentence: {sentence}"
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    label_id = int(logits.argmax().item())
    return model.config.id2label[label_id]


def load_canonical_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.input_jsonl.open("r", encoding="utf-8") as f_count:
        total_rows = sum(1 for _ in f_count)

    span_tokenizer = AutoTokenizer.from_pretrained(str(args.span_model_dir), use_fast=True)
    span_model = AutoModelForTokenClassification.from_pretrained(str(args.span_model_dir))
    span_model.eval()

    rel_tokenizer = AutoTokenizer.from_pretrained(str(args.relation_model_dir), use_fast=True)
    rel_model = AutoModelForSequenceClassification.from_pretrained(str(args.relation_model_dir))
    rel_model.eval()

    canon_map = load_canonical_map(args.canonical_map)

    with args.input_jsonl.open("r", encoding="utf-8") as fin, args.output_jsonl.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_rows, desc="inference:rows", unit="row"):
            row = json.loads(line)
            title = row.get("title", "")
            sentence = row.get("sentence") or row.get("sent", "")

            spans = predict_spans(sentence, span_tokenizer, span_model, max_length=args.max_length)
            for start, end, span_text in spans:
                rel = predict_relation(title, span_text, sentence, rel_tokenizer, rel_model, max_length=args.max_length)
                canonical = canon_map.get(rel, rel)
                out = {
                    "title": title,
                    "sentence": sentence,
                    "span_text": span_text,
                    "start": start,
                    "end": end,
                    "predicted_relation": rel,
                    "canonical_relation": canonical,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
