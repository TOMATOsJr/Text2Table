import argparse
import math
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight generation-only inference with a fine-tuned REBEL model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "inference_config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as err:
        raise ImportError(
            "PyYAML is required for config loading. Install with: pip install pyyaml"
        ) from err

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping/object")

    return config


def resolve_path(path_value: str, config_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


def require_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    section = cfg.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: '{key}'")
    return section


def get_required_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid required key: '{key}'")
    return value.strip()


def get_optional(section: dict[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("runtime.device='cuda' but CUDA is not available")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda")
    return torch.device(device_name)


def validate_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_torch_dtype(dtype_name: str, device: torch.device):
    if device.type != "cuda":
        return None
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "auto":
        # Ampere+ GPUs typically benefit from bf16; for RTX 2080 Ti use fp16.
        return torch.float16
    raise ValueError("runtime.torch_dtype must be one of: auto, fp16, bf16, fp32")


def iter_jsonl_records(
    input_path: Path,
    input_field: str,
    sample_size: int | None,
    skip_empty: bool,
):
    processed = 0
    skipped_invalid_json = 0
    skipped_missing_text = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid_json += 1
                continue

            if not isinstance(row, dict):
                skipped_invalid_json += 1
                continue

            raw_text = row.get(input_field)
            text = raw_text.strip() if isinstance(raw_text, str) else ""

            if not text:
                skipped_missing_text += 1
                if skip_empty:
                    continue

            yield {
                "row_index": line_num,
                "row": row,
                "text": text,
            }

            processed += 1
            if sample_size is not None and processed >= sample_size:
                break

    return skipped_invalid_json, skipped_missing_text


def chunked(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config_dir = args.config.resolve().parent

    model_cfg = require_section(config, "model")
    data_cfg = require_section(config, "data")
    runtime_cfg = require_section(config, "runtime")
    generation_cfg = require_section(config, "generation")
    output_cfg = require_section(config, "output")

    model_path = resolve_path(get_required_str(model_cfg, "model_path"), config_dir)
    input_path = resolve_path(get_required_str(data_cfg, "input_path"), config_dir)
    output_path = resolve_path(get_required_str(data_cfg, "output_path"), config_dir)

    input_field = str(get_optional(data_cfg, "input_field", "input_ids"))
    sample_size = get_optional(data_cfg, "sample_size", None)
    if sample_size is not None:
        sample_size = validate_positive_int("data.sample_size", sample_size)
    skip_empty = bool(get_optional(data_cfg, "skip_empty", True))

    batch_size = validate_positive_int(
        "runtime.batch_size", get_optional(runtime_cfg, "batch_size", 8)
    )
    device_name = str(get_optional(runtime_cfg, "device", "auto")).lower()
    multi_gpu = bool(get_optional(runtime_cfg, "multi_gpu", False))
    overwrite_output = bool(get_optional(runtime_cfg, "overwrite_output", True))
    torch_dtype_name = str(get_optional(runtime_cfg, "torch_dtype", "auto")).lower()

    max_input_length = validate_positive_int(
        "generation.max_input_length", get_optional(generation_cfg, "max_input_length", 512)
    )
    max_output_length = validate_positive_int(
        "generation.max_output_length", get_optional(generation_cfg, "max_output_length", 256)
    )
    num_beams = validate_positive_int(
        "generation.num_beams", get_optional(generation_cfg, "num_beams", 4)
    )
    num_return_sequences = validate_positive_int(
        "generation.num_return_sequences",
        get_optional(generation_cfg, "num_return_sequences", 1),
    )
    do_sample = bool(get_optional(generation_cfg, "do_sample", False))
    temperature = float(get_optional(generation_cfg, "temperature", 1.0))
    top_p = float(get_optional(generation_cfg, "top_p", 1.0))

    include_all_sequences = bool(get_optional(output_cfg, "include_all_sequences", False))
    include_metadata = bool(get_optional(output_cfg, "include_metadata", True))

    if num_return_sequences > num_beams and not do_sample:
        raise ValueError(
            "generation.num_return_sequences cannot exceed generation.num_beams when do_sample=false"
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(
            f"Output file already exists and overwrite_output=false: {output_path}"
        )

    device = resolve_device(device_name)
    torch_dtype = resolve_torch_dtype(torch_dtype_name, device)

    print(f"Loading tokenizer/model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)

    model_parallel_enabled = False
    if multi_gpu:
        if device.type != "cuda":
            raise ValueError("runtime.multi_gpu=true requires CUDA")
        if torch.cuda.device_count() < 2:
            print("runtime.multi_gpu=true but fewer than 2 GPUs are visible; using single GPU")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
            ).to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                device_map="balanced",
            )
            model_parallel_enabled = True
            print(f"Enabled model parallel inference on {torch.cuda.device_count()} GPUs")
            hf_map = getattr(model, "hf_device_map", None)
            if hf_map:
                used = sorted(
                    {
                        str(v)
                        for v in hf_map.values()
                        if isinstance(v, (int, str))
                    }
                )
                print(f"Model shards placed on devices: {', '.join(used)}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
        ).to(device)

    model.eval()

    total_candidates = 0
    total_written = 0

    # Materialize iterator stats separately because Python generators cannot
    # return counters in a straightforward, stream-friendly way.
    skipped_invalid_json = 0
    skipped_missing_text = 0

    def record_stream():
        nonlocal skipped_invalid_json, skipped_missing_text
        with input_path.open("r", encoding="utf-8") as f:
            processed = 0
            for line_num, raw_line in enumerate(
                tqdm(f, desc="Reading JSONL", unit="line", leave=False), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped_invalid_json += 1
                    continue

                if not isinstance(row, dict):
                    skipped_invalid_json += 1
                    continue

                raw_text = row.get(input_field)
                text = raw_text.strip() if isinstance(raw_text, str) else ""

                if not text:
                    skipped_missing_text += 1
                    if skip_empty:
                        continue

                yield {"row_index": line_num, "row": row, "text": text}
                processed += 1
                if sample_size is not None and processed >= sample_size:
                    break

    with output_path.open("w", encoding="utf-8") as writer:
        batch_total = None
        if sample_size is not None:
            batch_total = math.ceil(sample_size / batch_size)

        batch_iterator = tqdm(
            chunked(record_stream(), batch_size),
            desc="Generating",
            unit="batch",
            total=batch_total,
        )

        for batch in batch_iterator:
            batch_texts = [item["text"] for item in batch]
            total_candidates += len(batch_texts)

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )

            if not model_parallel_enabled:
                encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                generate_kwargs = {
                    "max_length": max_output_length,
                    "num_beams": num_beams,
                    "num_return_sequences": num_return_sequences,
                    "do_sample": do_sample,
                }
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p

                autocast_ctx = nullcontext()
                if device.type == "cuda" and torch_dtype in {torch.float16, torch.bfloat16}:
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch_dtype)

                with autocast_ctx:
                    generated = model.generate(
                        **encoded,
                        **generate_kwargs,
                    )

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)

            for i, item in enumerate(batch):
                start = i * num_return_sequences
                end = start + num_return_sequences
                sequences = [text.strip() for text in decoded[start:end]]

                output_row = dict(item["row"])
                output_row["generated_text"] = sequences[0] if sequences else ""

                if include_all_sequences:
                    output_row["generated_sequences"] = sequences

                if include_metadata:
                    output_row["inference_meta"] = {
                        "model_path": str(model_path),
                        "input_field": input_field,
                        "source_row_index": item["row_index"],
                        "num_return_sequences": num_return_sequences,
                        "num_beams": num_beams,
                        "multi_gpu": multi_gpu,
                        "gpu_count_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                        "torch_dtype": torch_dtype_name,
                    }

                writer.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                total_written += 1

            batch_iterator.set_postfix(rows=total_written, refresh=False)

    print("Inference completed")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Total processed datapoints: {total_candidates}")
    print(f"Total written rows: {total_written}")
    print(f"Skipped invalid JSON rows: {skipped_invalid_json}")
    print(f"Skipped rows with missing/empty '{input_field}': {skipped_missing_text}")


if __name__ == "__main__":
    main()
