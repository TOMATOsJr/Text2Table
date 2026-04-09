# REBEL Pipeline Inference (Lightweight)

This folder contains a generation-only inference pipeline for a fine-tuned REBEL model.

Scope of this script:
- Run model inference on JSONL input.
- Write raw generated output text.
- Do not parse generated text into triplets.

## Files

- `infer_rebel.py`: Config-driven inference entrypoint.
- `inference_config.yaml`: Runtime/config parameters.
- `rebel_to_kg.py`: Converts infer_rebel output into KG2Table-compatible JSON.

## Prerequisites

From workspace root (`/home2/jayanth.raju/rebel_dataset`):

```bash
uv pip install -r requirements.txt
```

Required assets:
- A fine-tuned REBEL model directory (default: `../../rebel_finetuned/best_model` from this folder).
- Input JSONL file with one object per line.

## Quick Start

Run with default config:

```bash
python3 rebel_pipeline/infer_rebel.py --config rebel_pipeline/inference_config.yaml
```

The default config uses:
- Model: `rebel_finetuned/best_model`
- Input: `test.jsonl`
- Output: `rebel_pipeline/predictions_test.jsonl`
- Datapoint limit: `sample_size: 25`

For full dataset inference, set in config:

```yaml
data:
  sample_size: null
```

## Input and Output Format

Input:
- JSONL with one JSON object per line.
- Script reads the field from `data.input_field` (default: `input_ids`).

Output:
- JSONL with original row preserved.
- Added field:
- `generated_text`: First generated sequence.
- Optional fields:
- `generated_sequences` when `output.include_all_sequences: true`.
- `inference_meta` when `output.include_metadata: true`.

## Config Reference (`inference_config.yaml`)

### model
- `model_path` (string, required): Fine-tuned model directory.

### data
- `input_path` (string, required): Input JSONL path.
- `output_path` (string, required): Output JSONL path.
- `input_field` (string, default `input_ids`): Source text field name.
- `sample_size` (int or null, default `25`): Max valid datapoints to process.
- `skip_empty` (bool, default `true`): Skip rows with missing/empty input text.

### runtime
- `batch_size` (int, default `8`): Inference batch size.
- `device` (`auto|cpu|cuda`, default `auto`): Device selection.
- `multi_gpu` (bool, default `false`): Use `device_map=balanced` when multiple GPUs are visible.
- `overwrite_output` (bool, default `true`): Overwrite output if it exists.

### generation
- `max_input_length` (int, default `512`)
- `max_output_length` (int, default `256`)
- `num_beams` (int, default `4`)
- `num_return_sequences` (int, default `1`)
- `do_sample` (bool, default `false`)
- `temperature` (float, default `1.0`, used only when `do_sample=true`)
- `top_p` (float, default `1.0`, used only when `do_sample=true`)

Constraint:
- If `do_sample=false`, then `num_return_sequences <= num_beams`.

### output
- `include_all_sequences` (bool, default `false`): Include all generated sequences.
- `include_metadata` (bool, default `true`): Include lightweight per-row metadata.

## Multi-GPU Notes

- `runtime.multi_gpu: true` requires CUDA and at least 2 visible GPUs.
- With multiple GPUs, model loading uses Hugging Face `device_map=balanced`.
- If fewer than 2 GPUs are visible, script falls back to single-GPU execution.

## Miscellaneous / Troubleshooting

- Error: `Config file not found`
- Fix: Verify `--config` path.

- Error: `Model path not found`
- Fix: Check `model.model_path` and confirm model files exist.

- Error: `Input dataset not found`
- Fix: Check `data.input_path`.

- Error: `overwrite_output=false`
- Fix: Delete/rename existing output or set `runtime.overwrite_output: true`.

- Error: `runtime.device='cuda' but CUDA is not available`
- Fix: Set `runtime.device: auto` or `cpu`, or fix CUDA environment.

- If output rows are fewer than input lines:
- Empty lines, invalid JSON rows, and missing input text rows can be skipped.
- Check end-of-run counters printed by the script.

## Design Decision

This script intentionally stays lightweight and single-responsibility:
- It performs generation only.
- It does not parse REBEL output into structured triplets.

## Convert for KG2Table

If you want to use `kg_to_table.py`, first convert raw inference output into the
expected schema (`record_id`, `title`, `sentence`, `entities`, `triples`):

```bash
python3 rebel_pipeline/rebel_to_kg.py \
  --input rebel_pipeline/predictions_test.jsonl \
  --output rebel_pipeline/processed_for_kg_to_table.json
```

Then run KG2Table:

```bash
python3 "Text2Table/Post mid sub files _ rough/kg_to_table.py" \
  --input rebel_pipeline/processed_for_kg_to_table.json \
  --output Text2Table/outputs/tables_from_rebel_pipeline.jsonl \
  --output-format jsonl
```
