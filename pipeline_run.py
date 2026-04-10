#!/usr/bin/env python3
"""
pipeline_run.py
===============
Full end-to-end Knowledge Graph Refinement Pipeline on 10,000 WikiBio samples.

Stages & Output Files
---------------------
  STAGE 0  – Build 10k sample from raw WikiBio .box/.sent files
              -> runs/s0_wikibio_sample.json

  STAGE 1  – Train QKV Relation Canonicalizer (from scratch on sample)
              -> runs/qkv_canon.pt

  STAGE 2  – Train QKV Entity Normalizer + build centroid index
              -> runs/qkv_enorm.pt

  STAGE 3  – Heuristic Refinement only (Steps 1-3: Inversion Fix, Snap, Injection)
              -> runs/s3_heuristic.json

  STAGE 4  – QKV Relation Canonicalization on top of Stage 3
              -> runs/s4_canon.json

  STAGE 5  – QKV Entity Normalization on top of Stage 4
              -> runs/s5_entity_norm.json

  STAGE 6  – Final Audit: stage-by-stage comparison table
              -> runs/audit_report.txt   (also printed to stdout)

Usage
-----
  ./.venv/bin/python3 pipeline_run.py [--force-retrain]

  --force-retrain  : Ignore existing checkpoints and retrain from scratch.
"""

import json
import re
import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────

WIKIBIO_DIR   = Path("/scratch/INLP_Project_Wiki/wikipedia-biography-dataset/train")
RUNS_DIR      = Path("runs")
N_SAMPLES     = 10_000
TRAIN_EPOCHS  = 15

QKV_CANON_CKPT = RUNS_DIR / "qkv_canon.pt"
QKV_ENORM_CKPT = RUNS_DIR / "qkv_enorm.pt"

S0_SAMPLE      = RUNS_DIR / "s0_wikibio_sample.json"
ENTITY_MERGES  = RUNS_DIR / "s0_entity_merges.json"
S3_HEURISTIC   = RUNS_DIR / "s3_heuristic.json"
S4_CANON       = RUNS_DIR / "s4_canon.json"
S5_ENTITY      = RUNS_DIR / "s5_entity_norm.json"
AUDIT_REPORT   = RUNS_DIR / "audit_report.txt"

# ─── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    mb = path.stat().st_size / 1_000_000
    log(f"  -> Saved {len(data):,} records to  {path}  ({mb:.1f} MB)")

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ─── STAGE 0: Build WikiBio Sample ────────────────────────────────────────────

def _parse_box(line: str) -> dict:
    """Parse one WikiBio .box line into {base_field: combined_value}."""
    raw = defaultdict(list)
    for tok in line.strip().split('\t'):
        if ':' not in tok:
            continue
        raw_key, _, val = tok.partition(':')
        if val == '<none>' or not val:
            continue
        base = re.sub(r'_\d+$', '', raw_key)
        raw[base].append(val)
    return {k: ' '.join(v) for k, v in raw.items()}


def build_wikibio_sample(n: int = N_SAMPLES) -> list[dict]:
    box_path  = WIKIBIO_DIR / "train.box"
    sent_path = WIKIBIO_DIR / "train.sent"

    log(f"STAGE 0 | Building {n:,} WikiBio records from raw files…")
    records = []

    with open(box_path, 'r', encoding='utf-8') as bf, \
         open(sent_path, 'r', encoding='utf-8') as sf:

        for idx, (box_line, sent_line) in enumerate(zip(bf, sf)):
            if idx >= n:
                break

            infobox   = _parse_box(box_line)
            sentences = [s.strip() for s in sent_line.strip().split('\t') if s.strip()]

            # Title from article_title field (multi-token stored separately)
            title = infobox.pop('article_title', '')
            if not title:
                title = infobox.get('name', f'person_{idx}').strip().title()
            else:
                title = title.strip().title()

            # Build raw triples directly from infobox key-value pairs
            SKIP_FIELDS = {'image', 'image_size', 'caption', 'footnotes',
                           'signature', 'website', 'medaltemplates', 'pcupdate',
                           'ntupdate', 'name'}
            raw_triples = []
            for field, value in infobox.items():
                if field in SKIP_FIELDS or not value.strip():
                    continue
                raw_triples.append({
                    "subject":  title,
                    "relation": field.replace('_', ' '),
                    "object":   value.strip(),
                })

            if not raw_triples:
                continue

            records.append({
                "title":     title,
                "sentences": sentences[:5],  # keep first 5 bio sentences
                "infobox":   infobox,
                "triples":   raw_triples,
            })

            if (idx + 1) % 2000 == 0:
                log(f"  Parsed {idx+1:,} / {n:,}")

    total_triples = sum(len(r['triples']) for r in records)
    log(f"  Done — {len(records):,} records, {total_triples:,} raw triples")
    return records


# ─── Build Entity Merge Clusters ──────────────────────────────────────────────

def build_entity_merges(records: list[dict]) -> list[dict]:
    """
    Build synonym clusters from Wikipedia infobox field values.
    These are used to train the contrastive entity normalizer.
    """
    log("  Building entity merge clusters…")
    clusters: dict[str, set] = {}

    def add(lead: str, syns: list[str]):
        lead = lead.strip()
        if len(lead) < 3:
            return
        if lead not in clusters:
            clusters[lead] = set()
        for s in syns:
            s = s.strip()
            if s and s != lead and len(s) > 2:
                clusters[lead].add(s)

    for r in records:
        title   = r['title']
        infobox = r.get('infobox', {})

        # Person name variants
        name = infobox.get('name', title)
        add(title, [name, title.lower(), name.upper()])

        # normalise birth_place variations
        bp = infobox.get('birth_place', '')
        if bp:
            parts = [p.strip() for p in re.split(r'[,/\-]', bp) if p.strip() and len(p.strip()) > 2]
            add(bp, parts)

        # nationality ↔ country
        nat = infobox.get('nationality', '')
        if nat:
            add(nat, [nat.lower(), nat.title()])

        # alma_mater / employer variants
        for field in ['alma_mater', 'employer', 'clubs']:
            val = infobox.get(field, '')
            if val:
                parts = [p.strip() for p in val.split() if len(p.strip()) > 3]
                add(val, parts)

    result = [{"lead": lead, "synonyms": list(syns)}
              for lead, syns in clusters.items() if syns]

    log(f"  Entity clusters built: {len(result):,}")
    return result


# ─── STAGE 3: Heuristic Refinement ────────────────────────────────────────────

INVERSION_RELS = {'member of', 'part of', 'played for', 'belongs to'}

def heuristic_refine(records: list[dict]) -> list[dict]:
    """
    Four classic heuristic steps:
    1. Relation Inversion Fix  – fix reversed (team→ member →person) triples
    2. Span Snapping           – snap truncated object text to full form
    3. Role Injection          – inject occupation/known_for if missing from triples
    4. Hallucination Filtering – remove name-as-occupation or person-as-place hallucinations
    """
    log("STAGE 3 | Heuristic Refinement (Inversion + Snap + Injection + Filtering)…")
    stats = {"inversions": 0, "snaps": 0, "injections": 0, "hallucinations": 0}
    refined = []

    for record in records:
        title   = record.get('title', '').lower()
        triples = [dict(t) for t in record.get('triples', [])]
        infobox = record.get('infobox', {})

        # Step 1 – Inversion Fix
        for t in triples:
            if t['relation'].lower() in INVERSION_RELS:
                if t['object'].strip().lower() == title:
                    t['subject'], t['object'] = t['object'], t['subject']
                    stats['inversions'] += 1

        # Step 2 – Span Snapping (snap partial objects to full entity spans)
        all_values = [e['text'] for e in record.get('entities', [])]
        for t in triples:
            obj_lower = t['object'].lower()
            for full_val in all_values:
                fv_lower = full_val.lower()
                if obj_lower != fv_lower and obj_lower in fv_lower and len(obj_lower) > 4:
                    t['object'] = full_val
                    stats['snaps'] += 1
                    break

        # Step 3 – Role Injection (From GLiNER Entities)
        triple_objects = {t['object'].lower() for t in triples}
        title_str = record.get('title', '')
        for e in record.get('entities', []):
            label = e.get('label', '').lower()
            text = e.get('text', '').strip()
            if label in ('profession', 'occupation', 'position', 'role') and text:
                if text.lower() not in triple_objects:
                    triples.append({
                        "subject": title_str,
                        "relation": "occupation" if label in ("profession", "role") else label,
                        "object": text,
                    })
                    stats['injections'] += 1
                    triple_objects.add(text.lower())

        # Step 4 – Hallucination Filtering (NEW)
        final_triples = []
        person_entities = {e['text'].lower() for e in record.get('entities', []) 
                           if e.get('label', '').upper() in ('PERSON', 'PER')}
        
        for t in triples:
            obj_lower = t['object'].lower()
            rel_lower = t['relation'].lower()
            
            # Criterion A: Subject-Object overlap hallucination
            if obj_lower == title or (len(obj_lower) > 5 and (obj_lower in title or title in obj_lower)):
                if rel_lower not in ('birth name', 'given name', 'family name', 'pseudonym'):
                    stats['hallucinations'] += 1
                    continue
            
            # Criterion B: Person-as-Place hallucination
            if 'place' in rel_lower or rel_lower in ('country', 'nationality', 'residence', 'location'):
                if obj_lower in person_entities:
                    stats['hallucinations'] += 1
                    continue
            
            final_triples.append(t)

        # Step 5 – Date Entity Snapping (NEW)
        # Often REBEL extracts "27" but spaCy correctly captures "27 February 1979"
        all_date_entities = [e['text'] for e in record.get('entities', []) 
                             if e.get('label', '').upper() == 'DATE']
        for t in final_triples:
            rel_low = t['relation'].lower()
            if 'date' in rel_low or 'birth' in rel_low or 'death' in rel_low or 'ended' in rel_low:
                obj_low = t['object'].lower()
                for de in all_date_entities:
                    de_low = de.lower()
                    if obj_low != de_low and obj_low in de_low and len(obj_low) > 1:
                        t['object'] = de
                        stats['snaps'] += 1
                        break

        new_rec = dict(record)
        new_rec['triples'] = final_triples
        refined.append(new_rec)

    log(f"  Inversions fixed: {stats['inversions']:,} | "
        f"Snaps: {stats['snaps']:,} | "
        f"Injections: {stats['injections']:,} | "
        f"Hallucinations Dropped: {stats['hallucinations']:,}")
    return refined


# ─── STAGE 6: Audit Report ────────────────────────────────────────────────────

def _audit_file(path: Path, label: str) -> dict:
    if not path.exists():
        return {"label": label, "missing": True}

    records = load_json(path)
    total_triples = 0
    raw_rels, canon_rels = Counter(), Counter()
    raw_ents, canon_ents = Counter(), Counter()
    n_canon_rel = 0
    n_canon_ent = 0

    for r in records:
        for t in r.get('triples', []):
            total_triples += 1

            rel = t.get('relation', '').lower()
            raw_rels[rel] += 1
            raw_ents[t.get('subject', '')] += 1
            raw_ents[t.get('object', '')] += 1

            if t.get('canonical_relation'):
                canon_rels[t['canonical_relation']] += 1
                n_canon_rel += 1

            cs = t.get('canonical_subject')
            co = t.get('canonical_object')
            if cs:
                canon_ents[cs] += 1
                n_canon_ent += 1
            else:
                canon_ents[t.get('subject', '')] += 1
            if co:
                canon_ents[co] += 1
                n_canon_ent += 1
            else:
                canon_ents[t.get('object', '')] += 1

    one_off_raw   = sum(1 for c in raw_rels.values() if c == 1)
    one_off_canon = sum(1 for c in canon_rels.values() if c == 1) if canon_rels else None

    return {
        "label":           label,
        "records":         len(records),
        "total_triples":   total_triples,
        "raw_unique_rels": len(raw_rels),
        "raw_one_off":     one_off_raw,
        "raw_sparsity":    round(one_off_raw / max(len(raw_rels), 1) * 100, 1),
        "canon_rels":      len(canon_rels) if canon_rels else "—",
        "canon_one_off":   one_off_canon if one_off_canon is not None else "—",
        "canon_pct":       round(n_canon_rel / max(total_triples, 1) * 100, 1),
        "raw_ents":        len(raw_ents),
        "canon_ents":      len(canon_ents),
        "ent_norm_pct":    round(n_canon_ent / max(total_triples * 2, 1) * 100, 1),
    }


def build_audit_report():
    stages = [
        (S0_SAMPLE,    "S0: Raw Infobox Triples"),
        (S3_HEURISTIC, "S3: + Heuristic Refinement"),
        (S4_CANON,     "S4: + QKV Relation Canon"),
        (S5_ENTITY,    "S5: + QKV Entity Norm"),
    ]
    log("STAGE 6 | Building Audit Report…")
    results = [_audit_file(p, l) for p, l in stages]

    lines = []
    SEP = "=" * 118
    lines += [SEP,
              "  KNOWLEDGE GRAPH PIPELINE — STAGE-BY-STAGE IMPACT ANALYSIS",
              SEP, ""]

    hdr = (f"{'Stage':<35} {'Records':>8} {'Triples':>9} "
           f"{'RawRels':>8} {'Sparsity':>9} "
           f"{'CanonRels':>10} {'CanonPct':>9} "
           f"{'RawEnts':>8} {'CanonEnts':>10} {'EntNorm%':>9}")
    lines += [hdr, "-" * 118]

    prev = None
    for m in results:
        if m.get('missing'):
            lines.append(f"  {m['label']} — FILE NOT FOUND")
            continue

        sparsity_delta = ""
        if prev and not prev.get('missing'):
            delta = m['raw_sparsity'] - prev['raw_sparsity']
            sparsity_delta = f"  ({delta:+.1f}%)"

        row = (f"{m['label']:<35} "
               f"{m['records']:>8,} "
               f"{m['total_triples']:>9,} "
               f"{m['raw_unique_rels']:>8} "
               f"{m['raw_sparsity']:>8.1f}%"
               f"{sparsity_delta:<12}"
               f"{str(m['canon_rels']):>10} "
               f"{m['canon_pct']:>8.1f}% "
               f"{m['raw_ents']:>8} "
               f"{m['canon_ents']:>10} "
               f"{m['ent_norm_pct']:>8.1f}%")
        lines.append(row)
        prev = m

    lines += ["",
              "LEGEND",
              "  Sparsity  = % of raw relations appearing exactly once (fragmentation / noise)",
              "  CanonRels = unique canonical schema slots mapped after QKV classification",
              "  CanonPct  = % of triples where QKV assigned a canonical relation",
              "  EntNorm%  = % of entity mentions mapped to a canonical cluster node",
              "",
              "WHAT GOOD LOOKS LIKE",
              "  S0→S3: Sparsity stays similar; triple count may increase (injections)",
              "  S3→S4: Sparsity drops sharply; CanonRels collapses 100s of noisy strings -> 57 slots",
              "  S4→S5: EntNorm% rises; CanonEnts < RawEnts (entity consolidation)",
              SEP]

    report = "\n".join(lines)
    print(report)

    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(report, encoding='utf-8')
    log(f"  Audit report saved -> {AUDIT_REPORT}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-retrain', action='store_true',
                        help='Ignore existing checkpoints and retrain QKV models from scratch.')
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log(f"  Full Pipeline Run — {N_SAMPLES:,} WikiBio samples")
    log("=" * 60)

    # ── Stage 0: WikiBio Sample ───────────────────────────────────
    if not S0_SAMPLE.exists():
        records = build_wikibio_sample(N_SAMPLES)
        save_json(records, S0_SAMPLE)
    else:
        log(f"STAGE 0 | Sample already exists — loading {S0_SAMPLE}")
        records = load_json(S0_SAMPLE)
        total0 = sum(len(r['triples']) for r in records)
        log(f"  Loaded {len(records):,} records, {total0:,} triples")

    # ── Entity Merges (training signal for Stage 2) ───────────────
    if not ENTITY_MERGES.exists():
        merges = build_entity_merges(records)
        save_json(merges, ENTITY_MERGES)
    else:
        log(f"  Entity merges exist — {ENTITY_MERGES}")

    # ── Stage 1: Train QKV Relation Canonicalizer ─────────────────
    if args.force_retrain or not QKV_CANON_CKPT.exists():
        log(f"STAGE 1 | Training QKV Relation Canonicalizer ({TRAIN_EPOCHS} epochs)…")
        from qkv_canonicalizer import QKVCanonicalizer, build_training_pairs
        canon = QKVCanonicalizer(schema_predicates=None)

        # Build training pairs from the sample with neural type classification (Fix 1)
        # Using cache_path to skip redundant neural classification across restarts (Fix 2)
        pair_result = build_training_pairs(
            input_json=str(S0_SAMPLE),
            schema_path=None,
            min_count=2,
            encoder_fn=canon._encode  # <-- Enable neural object-type resolve for training
        )
        train_pairs = pair_result['train_data']
        log(f"  Training pairs: {pair_result.get('rel_counts', {}).get('birth place', 0)} birth_place cases neural-typed.")
        log(f"  Total training pairs: {len(train_pairs):,}")

        log(f"  Starting Neural QKV Training ({TRAIN_EPOCHS} epochs)...")
        canon.train(
            train_data=train_pairs,
            epochs=TRAIN_EPOCHS,
            lr=3e-4,
            batch_size=256,
            eval_split=0.1,
        )
        canon.save(str(QKV_CANON_CKPT))
        log(f"  Training complete! Checkpoint saved to {QKV_CANON_CKPT}")
    else:
        log(f"STAGE 1 | Checkpoint exists — skipping training ({QKV_CANON_CKPT})")
        log("  (Use --force-retrain to retrain from scratch)")

    # ── Stage 2: Train QKV Entity Normalizer ─────────────────────
    if args.force_retrain or not QKV_ENORM_CKPT.exists():
        log(f"STAGE 2 | Training QKV Entity Normalizer ({TRAIN_EPOCHS} epochs)…")
        from qkv_canonicalizer import QKVEntityNormalizer, build_entity_training_pairs

        enorm = QKVEntityNormalizer()
        train_pairs = build_entity_training_pairs(
            entity_merges_json=str(ENTITY_MERGES),
            min_cluster_size=2,
        )
        log(f"  Entity training pairs: {len(train_pairs):,}")

        enorm.train(
            train_pairs=train_pairs,
            epochs=TRAIN_EPOCHS,
            lr=3e-4,
            batch_size=128,
            eval_split=0.1,
            verbose=True,
        )

        log("  Building centroid index…")
        enorm.build_centroid_index(entity_merges_json=str(ENTITY_MERGES))
        enorm.save(str(QKV_ENORM_CKPT))
        log(f"  Checkpoint + index -> {QKV_ENORM_CKPT}")
    else:
        log(f"STAGE 2 | Checkpoint exists — skipping training ({QKV_ENORM_CKPT})")

    # ── Stage 3: Heuristic Refinement ────────────────────────────
    log("STAGE 3 | Heuristic Refinement…")
    refined = heuristic_refine(records)
    save_json(refined, S3_HEURISTIC)

    # ── Stage 4: QKV Relation Canonicalization ───────────────────
    log("STAGE 4 | QKV Relation Canonicalization (threshold=0.10)…")
    from qkv_canonicalizer import QKVCanonicalizer
    canon2 = QKVCanonicalizer(schema_predicates=None)
    canon2.load(str(QKV_CANON_CKPT))
    canon2.canonicalize_kg(
        input_json=str(S3_HEURISTIC),
        output_json=str(S4_CANON),
        confidence_threshold=0.10,
    )
    log(f"  Stage 4 output -> {S4_CANON}")

    # ── Stage 5: QKV Entity Normalization ────────────────────────
    log("STAGE 5 | QKV Entity Normalization (threshold=0.75)…")
    from qkv_canonicalizer import QKVEntityNormalizer
    enorm2 = QKVEntityNormalizer()
    enorm2.load(str(QKV_ENORM_CKPT))
    enorm2.normalize_kg(
        input_json=str(S4_CANON),
        output_json=str(S5_ENTITY),
        similarity_threshold=0.75,
    )
    log(f"  Stage 5 output -> {S5_ENTITY}")

    # ── Stage 6: Audit Report ─────────────────────────────────────
    build_audit_report()

    log("=" * 60)
    log("Pipeline complete! All intermediate files saved to ./runs/")
    log(f"  s0_wikibio_sample.json  — Raw 10k infobox triples")
    log(f"  s3_heuristic.json       — After heuristic refinement")
    log(f"  s4_canon.json           — After QKV relation canonicalization")
    log(f"  s5_entity_norm.json     — After QKV entity normalization")
    log(f"  audit_report.txt        — Stage-by-stage comparison")
    log("=" * 60)


if __name__ == "__main__":
    main()
