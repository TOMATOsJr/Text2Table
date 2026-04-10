#!/usr/bin/env python3
"""Run REBEL output -> KG JSON -> KG2Table in one command.

This wrapper does two steps:
1) Convert infer_rebel output with rebel_to_kg.py
2) Build grouped attribute blocks with
    Text2Table/Post mid sub files _ rough/kg_to_table_csv.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list[str],
    label: str,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    print(f"\n[{label}] {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conversion and table generation as a single pipeline command."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./rebel_pipeline/predictions_grounded_test.jsonl"),
        help="Inference output file (JSONL/JSON) from infer_rebel",
    )
    parser.add_argument(
        "--kg-json",
        type=Path,
        default=Path("./rebel_pipeline/processed_for_grounded_kg_to_table.json"),
        help="Intermediate KG-compatible JSON output path",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("./rebel_pipeline/tables/tables_from_rebel_grounded_pipeline.txt"),
        help="Final grouped table output path",
    )
    parser.add_argument(
        "--convert-limit",
        type=int,
        default=0,
        help="Limit rows for rebel_to_kg.py (0 = all)",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=0,
        help="Limit records for kg_to_table_csv.py (0 = all)",
    )
    parser.add_argument(
        "--no-normalize-relations",
        action="store_true",
        help="Disable relation normalization in kg_to_table_csv.py",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    convert_script = repo_root / "rebel_pipeline" / "rebel_to_kg.py"
    kg_to_table_script = repo_root / "Post mid sub files _ rough" / "kg_to_table_csv.py"

    input_path = (repo_root / args.input).resolve() if not args.input.is_absolute() else args.input
    kg_json_path = (
        (repo_root / args.kg_json).resolve() if not args.kg_json.is_absolute() else args.kg_json
    )
    table_output_path = (
        (repo_root / args.table_output).resolve()
        if not args.table_output.is_absolute()
        else args.table_output
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not convert_script.exists():
        raise FileNotFoundError(f"Converter script not found: {convert_script}")
    if not kg_to_table_script.exists():
        raise FileNotFoundError(f"KG2Table script not found: {kg_to_table_script}")

    kg_json_path.parent.mkdir(parents=True, exist_ok=True)
    table_output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_cmd = [
        args.python,
        str(convert_script),
        "--input",
        str(input_path),
        "--output",
        str(kg_json_path),
    ]
    if args.convert_limit > 0:
        convert_cmd.extend(["--limit", str(args.convert_limit)])

    table_cmd = [
        args.python,
        str(kg_to_table_script),
        "--input",
        str(kg_json_path),
        "--output",
        str(table_output_path),
    ]
    if args.table_limit > 0:
        table_cmd.extend(["--limit", str(args.table_limit)])
    if args.no_normalize_relations:
        table_cmd.append("--no-normalize-relations")

    print("Pipeline start")
    print(f"Input: {input_path}")
    print(f"Intermediate KG JSON: {kg_json_path}")
    print(f"Final table output: {table_output_path}")

    run_command(convert_cmd, "convert")
    run_command(
        table_cmd,
        "table",
        cwd=repo_root,
        extra_env={"PYTHONPATH": str(repo_root)},
    )

    print("\nPipeline complete")


if __name__ == "__main__":
    main()
