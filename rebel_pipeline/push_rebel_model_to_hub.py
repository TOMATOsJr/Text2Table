#!/usr/bin/env python3
"""Upload the fine-tuned REBEL best_model folder to a Hugging Face model repo.

This uploads the full folder so model.safetensors, tokenizer files, and config
files are all versioned in the target Hub repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

# Global macros requested by user.
HUB_ID = "TOMATOsJr/Rebel_Finetuned_on_Triples"
PUSH_TOKEN = ""

# Default local model directory.
MODEL_DIR = Path(__file__).resolve().parent / "rebel_finetuned" / "best_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push local fine-tuned REBEL best_model to Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Path to local best_model folder",
    )
    parser.add_argument(
        "--hub-id",
        type=str,
        default=HUB_ID,
        help="Target Hub repo id, e.g. username/repo-name",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=PUSH_TOKEN,
        help="Hugging Face write token. If empty, HF_TOKEN env var is used.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload fine-tuned REBEL best_model",
        help="Commit message used for upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    hub_id = args.hub_id.strip()
    token = args.token.strip() or os.getenv("HF_TOKEN", "").strip()

    if not hub_id or "/" not in hub_id:
        raise ValueError(
            "Invalid hub id. Set HUB_ID or pass --hub-id as username/repo-name"
        )

    if not token:
        raise ValueError(
            "Missing token. Set PUSH_TOKEN or HF_TOKEN env var, or pass --token"
        )

    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Expected model weights file missing: {weights_path}"
        )

    api = HfApi(token=token)

    api.create_repo(
        repo_id=hub_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    commit_info = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=hub_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print(f"Uploaded model folder: {model_dir}")
    print(f"Repo: https://huggingface.co/{hub_id}")
    print(f"Commit: {commit_info.oid}")


if __name__ == "__main__":
    main()