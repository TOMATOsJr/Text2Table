#!/usr/bin/env python3
"""Train relation classifier with LUKE (or any sequence classification model).

Example (4 GPUs):
torchrun --nproc_per_node=4 scripts/train_relation_luke.py \
  --train_file processed/relation_cls_train.jsonl \
  --valid_file processed/relation_cls_valid.jsonl \
  --model_name studio-ousia/luke-base \
  --output_dir outputs/relation_luke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--valid_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="studio-ousia/luke-base")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/relation_classifier"))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    return parser.parse_args()


def format_input(title: str, span_text: str, sentence: str) -> str:
    return f"title: {title} [SEP] span: {span_text} [SEP] sentence: {sentence}"


def collect_labels(train_file: Path, valid_file: Path) -> List[str]:
    labels = set()
    for fp in (train_file, valid_file):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                labels.add(row["relation"])
    return sorted(labels)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_list = collect_labels(args.train_file, args.valid_file)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    raw = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.valid_file)},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    def preprocess(example: Dict) -> Dict:
        text = format_input(example["title"], example["span_text"], example["sentence"])
        enc = tokenizer(text, truncation=True, max_length=args.max_length)
        enc["labels"] = label2id[example["relation"]]
        return enc

    tokenized = raw.map(preprocess, remove_columns=raw["train"].column_names)

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
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
    tokenizer.save_pretrained(str(args.output_dir / "best"))

    with (args.output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": label_list, "label2id": label2id, "id2label": id2label}, f, indent=2)


if __name__ == "__main__":
    main()
