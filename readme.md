# Biographical Knowledge Graph Pipeline: Performance Report

This repository contains the end-to-end pipeline for extracting high-fidelity structured Knowledge Graphs from the WikiBio dataset.

## Pipeline Architecture
- **Stage 1**: Hybrid REBEL + GLiNER Extraction.
- **Stage 2**: Spacy-based Syntactic Grounding.
- **Stage 3**: Heuristic Refinement (Inversion fixing, Hallucination filtering, and Date Snap recovery).
- **Stage 4 & 5 (Optional)**: Neural QKV Canonicalization & Entity Normalization.

---

## Comparative Metrics (1k WikiBio Test Sample)

The following metrics compare the pipeline with and without the Neural Canonicalization (Stages 4 & 5). Results indicate that the Raw Heuristic pipeline is significantly more accurate for this specific biography domain.

| Metric | Baseline (Heuristics Only) | With Canonicalisation (QKV) | Change |
| :--- | :---: | :---: | :---: |
| **chrF Score** (Structure) | **42.77** | 29.35 | 🔻 -13.42 |
| **Triple Precision** | **0.2603** | 0.0531 | 🔻 -80.0% |
| **Triple Recall** | **0.2705** | 0.0506 | 🔻 -81.0% |
| **Triple F1** | **0.2653** | **0.0518** | 🔻 -80.5% |

### Root Cause Analysis: Neural Drift
During the audit of the "With Canonicalisation" run, we identified significant **Neural Drift** in the QKV classification models:
1. **Misclassification**: Obvious relations like `date of birth` were incorrectly mapped to standard schema labels like `member of political party` (86% confidence), leading to catastrophic label mismatch during evaluation.
2. **Hallucinated Injections**: The canonicalization models frequently injected unrelated string artifacts from the training distribution into the extraction fields.

**Recommendation**: Use the **Heuristics Only** pipeline configuration for large-scale WikiBio production runs.

---

## Usage
To reproduce the peak performance metrics (Baseline):
```bash
python3 run_fixed_1k_pipeline.py
```
*(Note: Canonicalization logic remains in code for architectural reference but is bypassed by default in the reinforced table generation path.)*
