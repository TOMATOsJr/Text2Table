import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune REBEL on JSONL triplet data using 4 GPUs with torchrun."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("pre-processing/rebel_dataset/train.jsonl"),
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--valid-file",
        type=Path,
        default=Path("pre-processing/rebel_dataset/valid.jsonl"),
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("pre-processing/rebel_dataset/test.jsonl"),
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Babelscape/rebel-large",
        help="Base model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pre-processing/rebel_finetuned"),
        help="Root output directory for checkpoints and final best model",
    )
    parser.add_argument(
        "--best-model-dir",
        type=Path,
        default=None,
        help="Directory to write the final best model. Defaults to <output-dir>/best_model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=40.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--require-world-size",
        type=int,
        default=4,
        help="Require this WORLD_SIZE to ensure all GPUs are used.",
    )
    return parser.parse_args()


def check_runtime(require_world_size: int) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != require_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but this run requires {require_world_size}. "
            f"Launch with: torchrun --nproc_per_node={require_world_size} ..."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU training requires CUDA.")

    gpu_count = torch.cuda.device_count()
    if gpu_count < require_world_size:
        raise RuntimeError(
            f"Found {gpu_count} visible CUDA devices, need at least {require_world_size}."
        )


def load_json_dataset(train_file: Path, valid_file: Path, test_file: Path):
    for p in (train_file, valid_file, test_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    return load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(valid_file),
            "test": str(test_file),
        },
    )


def main() -> None:
    args = parse_args()
    check_runtime(args.require_world_size)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    best_model_dir = args.best_model_dir or (args.output_dir / "best_model")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    raw_ds = load_json_dataset(args.train_file, args.valid_file, args.test_file)

    def preprocess_batch(batch):
        model_inputs = tokenizer(
            batch["input_ids"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["labels"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing dataset",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_dataset = cast(Any, tokenized["train"])
    validation_dataset = cast(Any, tokenized["validation"])
    test_dataset = cast(Any, tokenized["test"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(best_model_dir)

    val_metrics = trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="valid")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    # Keep only the final best model artifact.
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    metrics_path = best_model_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_result.metrics,
                "valid": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved best model and tokenizer to: {best_model_dir}")
    print(f"Saved metrics to: {metrics_path}")
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
