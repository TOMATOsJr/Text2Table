#!/usr/bin/env python3
"""Train span extractor as token classification (BIO tags).

Example (4 GPUs):
torchrun --nproc_per_node=4 scripts/train_span_extractor.py \
  --train_file processed/span_ner_train.jsonl \
  --valid_file processed/span_ner_valid.jsonl \
  --model_name bert-base-uncased \
  --output_dir outputs/span_bert

For SpanBERT, try model_name=SpanBERT/spanbert-base-cased.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABELS = ["O", "B-ATTR", "I-ATTR"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--valid_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/span_extractor"))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    return parser.parse_args()


def tokenize_and_align_labels(example: Dict, tokenizer, max_length: int) -> Dict:
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    word_ids = tokenized.word_ids()
    labels: List[int] = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(LABEL2ID.get(example["ner_tags"][word_idx], LABEL2ID["O"]))
        else:
            # Ignore subword continuation positions in loss.
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.valid_file)},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenized = raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, args.max_length),
        remove_columns=raw["train"].column_names,
        batched=False,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=100,
        weight_decay=0.01,
        fp16=args.fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
    tokenizer.save_pretrained(str(args.output_dir / "best"))


if __name__ == "__main__":
    main()
