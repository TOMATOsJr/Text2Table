"""
Phase 6 – Knowledge Graph to Table Conversion
===============================================

Converts extracted triples (from the hybrid REBEL + GLiNER extraction pipeline)
into structured WikiBio-style attribute tables.

This script is the missing Stage 6 of the Text2Table pipeline described in
plan.txt: "Convert the knowledge graph into a structured table."

Core challenges addressed (per team discussions):
  1. Anchor detection     – Which triples belong to the biography subject?
                            ("how do you find what the anchor is?")
  2. Relation normalisation – Map free-text relations to canonical attributes
                            (Vishwas's relation normalisation task)
  3. Date/place disambiguation – "born on" (time) vs "born at" (place)
                            (Tulasi's POS-tagging idea, implemented via entity types)
  4. Deduplication          – Remove redundant triples and values
                            ("redundant triplets, too many rows")
  5. Single-column tables   – "simplify into one column only" (Jayanth's point)

References:
  - Kertkeidkachorn & Ichise (2018): T2KG framework — hybrid rule + similarity
    predicate mapping. Our normalisation follows the same philosophy.
  - plan.txt: Full 6-stage pipeline specification with examples.

Input:
  processed_triples_44k_refined.json  — JSON array of per-sentence extractions,
  each with record_id, title, sentence, entities[], triples[].
  NOTE: We use the "refined" version, NOT "resolved" which introduces an object-
  duplication regression (e.g. "baseball major league baseball").

Output (one of three formats):
  - JSONL  (default)  : One JSON per line with structured table dict
  - .box   (WikiBio)  : Tab-separated indexed tokens for BLEU evaluation
  - .md    (Markdown)  : Human-readable tables for manual evaluation

Usage:
  python kg_to_table.py \\
      --input "Post mid sub files _ rough/processed_triples_44k_refined.json" \\
      --output outputs/tables.jsonl

  python kg_to_table.py \\
      --input "Post mid sub files _ rough/processed_triples_44k_refined.json" \\
      --output outputs/tables.md --output-format markdown --limit 20

  python kg_to_table.py \\
      --input "Post mid sub files _ rough/processed_triples_44k_refined.json" \\
      --output outputs/tables.box --output-format box
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import relation mapping from existing pipeline module (postprocess.py).
# postprocess.py only uses standard-lib so this is a lightweight import.
# If you run from a different directory, add the project root to sys.path.
# ---------------------------------------------------------------------------
try:
    from postprocess import RELATION_TO_ATTRIBUTE, normalise_relation, _auto_normalise
except ImportError:
    # Fallback: add project root to path and retry
    _project_root = str(Path(__file__).resolve().parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from postprocess import RELATION_TO_ATTRIBUTE, normalise_relation, _auto_normalise


# ============================================================================
#  CONSTANTS
# ============================================================================

# --- Seed relations (duplicated from relation_extraction.py) ----------------
# We duplicate rather than import because relation_extraction.py pulls in heavy
# dependencies (torch, transformers, spacy) that are only needed at extraction
# time, not at table-generation time.

SEED_RELATIONS: set[str] = {
    "date of birth", "place of birth", "country of citizenship",
    "date of death", "place of death", "cause of death",
    "occupation", "position held", "employer", "work location",
    "member of sports team", "sport", "position played on team",
    "educated at", "alma mater", "academic degree",
    "spouse", "child", "father", "mother", "sibling", "relative",
    "award received", "nominated for", "member of",
    "genre", "instrument", "record label", "performer",
    "military rank", "conflict", "member of political party",
    "country", "nationality", "place of burial", "residence",
    "notable work", "author", "publisher", "language of work or name",
    "participant of", "coach of sports team", "religion",
    "country for sport", "participant in",
    "languages spoken written or signed",
    "field of work", "instance of", "sex or gender",
    "different from", "parent organization", "subsidiary", "inception",
    "followed by", "follows", "part of", "has part", "subclass of",
    "sports season of league or competition", "point in time",
    "located in the administrative territorial entity",
    "contains administrative territorial entity",
    "applies to jurisdiction", "legislated by",
}

# --- Attribute cardinality --------------------------------------------------
# Single-valued attributes have exactly one correct value per person.
# Multi-valued attributes (clubs, awards, …) can have many.
# Anything not in either set defaults to multi-valued.

SINGLE_VALUED_ATTRS: set[str] = {
    "name", "fullname", "birth_name",
    "birth_date", "birth_place",
    "death_date", "death_place", "death_cause",
    "gender", "nationality", "religion",
}

# --- Canonical display order ------------------------------------------------
# Attributes appear in this order in the output table.  Unlisted attributes
# are appended alphabetically after these.

ATTRIBUTE_ORDER: list[str] = [
    "name", "fullname", "birth_name",
    "birth_date", "birth_place",
    "death_date", "death_place", "death_cause",
    "nationality", "gender", "religion",
    "occupation", "position", "office",
    "alma_mater", "education",
    "sport", "clubs", "team",
    "party", "member_of",
    "spouse", "children", "parents", "relations",
    "awards", "known_for", "notable_work",
    "genre", "instrument", "label", "associated_acts",
    "residence", "country", "location",
    "year",
]

# --- Entity-type label sets (for date/place disambiguation) -----------------
# These come from spaCy NER + GLiNER labels present in the input JSON.

LOCATION_LABELS: set[str] = {
    "GPE", "LOC", "FAC",                     # spaCy NER
    "gpe", "loc", "fac",                     # lowercase variants
    "location", "place", "city", "country",  # GLiNER custom labels
}

DATE_LABELS: set[str] = {
    "DATE", "TIME",                          # spaCy NER
    "date", "time",                          # lowercase variants
    "CARDINAL", "ORDINAL",                   # sometimes dates appear as these
}

# Minimum token length to count as "non-trivial" during anchor matching.
# Avoids false positives on words like "a", "of", "in", "is".
MIN_ANCHOR_TOKEN_LEN = 3

# Structural / overly-generic relations to drop.  These create noise in
# biography tables (e.g. "instance_of: department", "subclass_of: strategy").
# Identified by inspecting the current records.jsonl output.
DROP_ATTRIBUTES: set[str] = {
    "instance_of", "subclass_of", "different_from",
    "part_of", "has_part",
    "follows", "followed_by",
    "applies_to_jurisdiction", "legislated_by",
    "contains_administrative_territorial_entity",
    "located_in_the_administrative_territorial_entity",
}


# ============================================================================
#  PHASE 1 — Data Loading & Grouping
# ============================================================================

def load_and_group(input_path: Path) -> dict[int, dict]:
    """Load processed triples JSON and group sentence-level entries by record.

    The input JSON is an array where each element represents ONE sentence from
    ONE biography.  Multiple elements share the same record_id.  This function
    merges them into per-person records.

    Returns
    -------
    {record_id: {
        "title":     str,          # person name (biography subject)
        "triples":   list[dict],   # all triples across all sentences
        "entities":  list[dict],   # all entities across all sentences
        "sentences": list[str],    # unique source sentences
    }}
    """
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records: dict[int, dict] = {}

    skipped = 0
    for entry in raw_data:
        # Skip entries with empty titles — these are extraction artefacts
        # (10K+ entries dumped into record_id=0 with no subject identity).
        title = entry.get("title", "").strip()
        if not title:
            skipped += 1
            continue

        rid = entry["record_id"]

        if rid not in records:
            records[rid] = {
                "title": entry.get("title", ""),
                "triples": [],
                "entities": [],
                "sentences": [],
            }

        rec = records[rid]

        # Accumulate triples produced by REBEL / GLiNER for this sentence.
        for triple in entry.get("triples", []):
            rec["triples"].append(triple)

        # Accumulate named entities (used later for type-based disambiguation).
        for entity in entry.get("entities", []):
            rec["entities"].append(entity)

        # Track unique source sentences (avoids duplicates from overlapping
        # extractions when the same sentence appears in multiple entries).
        sentence = entry.get("sentence", "")
        if sentence and sentence not in rec["sentences"]:
            rec["sentences"].append(sentence)

    if skipped:
        print(f"  Skipped {skipped:,} entries with empty titles")

    return records


# ============================================================================
#  PHASE 2 — Anchor Detection & Subject Filtering
# ============================================================================
#
# The "anchor" is the person the biography is about.  The team discussed this:
#   "if a paragraph is given about Tulsi, how would I know the paragraph is
#    about Tulsi only?  It's not about his parents or his friends"
#
# Our answer: the title field identifies the subject.  We build an alias set
# from the title + all PERSON entities, then keep only triples whose subject
# token-overlaps with that alias set.
# ============================================================================

def build_alias_set(title: str, entities: list[dict]) -> set[str]:
    """Build a set of lowercase name aliases for the biography subject.

    Strategy
    --------
    1. Start with the ``title`` (canonical name from the dataset).
    2. Add every entity whose label is PERSON — these are name variations
       the NER model found in the text (e.g. full name, maiden name).
    3. Add individual name tokens ≥ 4 chars from the above.  This handles
       REBEL's tendency to output partial names (e.g. "shenoff randle"
       instead of "leonard shenoff randle").

    Example
    -------
    title = "lenny randle"
    PERSON entities = ["leonard shenoff randle"]
    → aliases = {"lenny randle", "leonard shenoff randle",
                  "lenny", "randle", "leonard", "shenoff"}
    """
    aliases: set[str] = set()

    # 1. Title is always an alias.
    title_lower = title.strip().lower()
    if title_lower:
        aliases.add(title_lower)

    # 2. All PERSON entities are potential aliases.
    for ent in entities:
        if ent.get("label", "").upper() == "PERSON":
            ent_text = ent["text"].strip().lower()
            if ent_text:
                aliases.add(ent_text)

    # 3. Individual tokens for partial-match support.
    #    Only tokens ≥ 4 chars to avoid "a", "of", "the", etc.
    token_aliases: set[str] = set()
    for alias in list(aliases):                     # iterate over a snapshot
        for token in alias.split():
            if len(token) >= 4:
                token_aliases.add(token)
    aliases.update(token_aliases)

    return aliases


def subject_matches_anchor(subject: str, aliases: set[str]) -> bool:
    """Decide whether a triple's subject refers to the biography's person.

    Match criteria (in order):
      1. Exact match against any alias string.
      2. At least one non-trivial token (≥ 3 chars) shared with the alias set.

    This handles:
      - Exact: "lenny randle" ∈ aliases                          → True
      - Partial: "shenoff randle" shares "randle" with aliases   → True
      - Unrelated: "major league" shares nothing meaningful       → False
    """
    subj_lower = subject.strip().lower()

    # Exact match
    if subj_lower in aliases:
        return True

    # Token overlap
    for token in subj_lower.split():
        if len(token) >= MIN_ANCHOR_TOKEN_LEN and token in aliases:
            return True

    return False


def filter_triples_by_anchor(
    triples: list[dict],
    aliases: set[str],
) -> list[dict]:
    """Keep only triples whose subject refers to the biography's person.

    This is the critical filter that addresses:
      "how do we differentiate between information that is related to the
       original person and other persons who are related to you?"

    Triples about other entities (subject = "major league",
    subject = "washington senators") are dropped.
    """
    return [t for t in triples if subject_matches_anchor(t["subject"], aliases)]


# ============================================================================
#  PHASE 3 — Relation Normalisation
# ============================================================================
#
# Three sub-steps:
#   3a. Split concatenated relations  (REBEL artefact)
#   3b. Map relation → canonical attribute key
#   3c. Disambiguate date vs place using entity types
# ============================================================================

# --- 3a. Concatenated relation splitting ------------------------------------

def split_concatenated_relation(relation: str) -> list[str]:
    """Split a concatenated relation string into individual known relations.

    REBEL sometimes merges two relations into one string, e.g.:
        "date of birth place of birth"
    This function applies greedy left-to-right matching against SEED_RELATIONS
    to recover the individual relations:
        → ["date of birth", "place of birth"]

    If the string is already atomic or no valid split is found, returns it
    unchanged in a single-element list.

    Based on ``split_concatenated_triples()`` in relation_extraction.py.
    """
    rel = relation.strip().lower()

    # Already a known atomic relation — no split needed.
    if rel in SEED_RELATIONS or rel in RELATION_TO_ATTRIBUTE:
        return [rel]

    # Greedy left-to-right matching (longest match first).
    sorted_rels = sorted(SEED_RELATIONS, key=len, reverse=True)
    found: list[str] = []
    remaining = rel

    while remaining:
        matched = False
        for known in sorted_rels:
            if remaining.startswith(known):
                found.append(known)
                remaining = remaining[len(known):].strip()
                matched = True
                break
        if not matched:
            break

    # Accept the split only if we consumed the ENTIRE string and found ≥ 2
    # relations.  Otherwise treat it as a single (possibly unknown) relation.
    if len(found) >= 2 and remaining.strip() == "":
        return found

    return [rel]


# --- 3c. Entity-type disambiguation ----------------------------------------

def disambiguate_date_place(
    attribute: str,
    value: str,
    entity_lookup: dict[str, str],
) -> str:
    """Swap date↔place attributes when entity types indicate a mismatch.

    Addresses Tulasi's discussion about "born on" (time) vs "born at" (place).
    Instead of POS-tagging the relation string, we check the entity type of the
    VALUE.  The entities are already extracted by GLiNER / spaCy, so this costs
    nothing.

    Example corrections:
      attribute="birth_date", value="rhèges" → entity is GPE → "birth_place"
      attribute="birth_place", value="1949"  → entity is DATE → "birth_date"

    Parameters
    ----------
    attribute     : Current attribute key.
    value         : The triple's object value.
    entity_lookup : {lowercase_entity_text: uppercase_label} built once per record.
    """
    date_attrs  = {"birth_date", "death_date"}
    place_attrs = {"birth_place", "death_place"}

    if attribute not in date_attrs and attribute not in place_attrs:
        return attribute

    value_lower = value.strip().lower()

    for ent_text, ent_label in entity_lookup.items():
        # Check if the value matches (or is contained in) this entity.
        if ent_text in value_lower or value_lower in ent_text:
            if attribute in date_attrs and ent_label in LOCATION_LABELS:
                return attribute.replace("date", "place")
            if attribute in place_attrs and ent_label in DATE_LABELS:
                return attribute.replace("place", "date")

    return attribute


# --- Combined normalisation -------------------------------------------------

def build_entity_lookup(entities: list[dict]) -> dict[str, str]:
    """Build a fast {text → label} lookup for disambiguation.

    Lowercases entity text for case-insensitive matching.  If the same text
    appears with multiple labels, the last one wins (rare edge case).
    """
    lookup: dict[str, str] = {}
    for ent in entities:
        text = ent.get("text", "").strip().lower()
        label = ent.get("label", "").upper()
        if text and label:
            lookup[text] = label
    return lookup


def normalise_triples(
    triples: list[dict],
    entity_lookup: dict[str, str],
) -> list[tuple[str, str]]:
    """Convert filtered triples into (attribute, value) pairs.

    Pipeline per triple:
      1. Split concatenated relations  →  may yield multiple (rel, obj) pairs
      2. Map each relation to a canonical attribute via ``normalise_relation()``
      3. Disambiguate date/place via entity types
      4. Drop structural / non-informative attributes (DROP_ATTRIBUTES)

    Returns a list of (attribute_key, value_string) tuples.
    """
    pairs: list[tuple[str, str]] = []

    for triple in triples:
        relation = triple["relation"].strip().lower()
        value    = triple["object"].strip()

        if not value:
            continue

        # 3a – Attempt to split concatenated relations.
        split_rels = split_concatenated_relation(relation)

        for rel in split_rels:
            # 3b – Canonical attribute mapping.
            attr = normalise_relation(rel)

            # 3c – Entity-type disambiguation (date ↔ place).
            attr = disambiguate_date_place(attr, value, entity_lookup)

            # Drop structural attributes that pollute biography tables.
            if attr in DROP_ATTRIBUTES:
                continue

            pairs.append((attr, value))

    return pairs


# ============================================================================
#  PHASE 4 — Deduplication & Value Aggregation
# ============================================================================

def deduplicate_values(values: list[str]) -> list[str]:
    """Remove duplicates and redundant substrings within an attribute's values.

    Two strategies applied sequentially:

    1. **Case-insensitive exact dedup** — "Baseball" and "baseball" are treated
       as the same; the first occurrence's casing is kept.

    2. **Substring absorption** — If "baseball" and "major league baseball"
       both appear, the shorter one is absorbed by the longer one.  This fixes
       the duplication bug observed in processed_triples_44k_resolved.json
       where objects were erroneously concatenated.

    Example
    -------
    ["baseball", "major league baseball", "Baseball"]
      → ["major league baseball"]
    """
    if not values:
        return values

    # Step 1: Case-insensitive exact dedup (preserve first casing).
    seen_lower: set[str] = set()
    unique: list[str] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        v_lower = v.lower()
        if v_lower not in seen_lower:
            seen_lower.add(v_lower)
            unique.append(v)

    if len(unique) <= 1:
        return unique

    # Step 2: Substring absorption — sort longest first, drop shorter values
    # that are fully contained in a longer already-accepted value.
    sorted_by_len = sorted(unique, key=len, reverse=True)
    kept: list[str] = []

    for val in sorted_by_len:
        val_lower = val.lower()
        is_absorbed = any(
            val_lower in accepted.lower() and val_lower != accepted.lower()
            for accepted in kept
        )
        if not is_absorbed:
            kept.append(val)

    return kept


def pick_best_single_value(values: list[str]) -> str:
    """For single-valued attributes, pick the most informative value.

    Heuristic: longest string wins, because it usually has the most complete
    information (e.g. "february 12, 1949" > "1949").
    """
    if not values:
        return ""
    return max(values, key=len)


def aggregate_attributes(
    attr_value_pairs: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Group (attribute, value) pairs into a table dict.

    - Single-valued attributes (birth_date, nationality, …) → keep only the
      most informative value.
    - Multi-valued attributes (clubs, awards, …) → keep all unique values.
    - Unknown attributes → default to multi-valued.

    Returns {attribute: [value, ...]} with deduplication applied.
    """
    # Group all values by attribute.
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for attr, value in attr_value_pairs:
        grouped[attr].append(value)

    result: dict[str, list[str]] = {}
    for attr, vals in grouped.items():
        deduped = deduplicate_values(vals)
        if not deduped:
            continue

        if attr in SINGLE_VALUED_ATTRS:
            result[attr] = [pick_best_single_value(deduped)]
        else:
            result[attr] = deduped

    return result


# ============================================================================
#  PHASE 5 — Table Assembly
# ============================================================================

def sort_attributes(
    attributes: dict[str, list[str]],
) -> list[tuple[str, list[str]]]:
    """Sort attribute keys into canonical display order.

    Attributes in ATTRIBUTE_ORDER appear first (in that order).  Remaining
    attributes are appended alphabetically.  This guarantees that every table
    starts with name → birth → death → personal → career → other.
    """
    ordered: list[tuple[str, list[str]]] = []
    remaining = set(attributes.keys())

    for attr in ATTRIBUTE_ORDER:
        if attr in attributes:
            ordered.append((attr, attributes[attr]))
            remaining.discard(attr)

    for attr in sorted(remaining):
        ordered.append((attr, attributes[attr]))

    return ordered


def build_table(record: dict) -> dict[str, Any]:
    """Run the full KG → Table pipeline for one biography record.

    Stages executed:
      2. Anchor detection & subject filtering
      3. Relation normalisation (split + map + disambiguate)
      4. Value deduplication & aggregation
      5. Table assembly with canonical ordering

    Parameters
    ----------
    record : dict with keys "title", "triples", "entities", "sentences".

    Returns
    -------
    {
      "title":                str,
      "num_attributes":       int,
      "num_triples_input":    int,   # before filtering
      "num_triples_filtered": int,   # after anchor filter
      "table":  [(attr, [values]), ...]   # ordered
    }
    """
    title    = record["title"]
    triples  = record["triples"]
    entities = record["entities"]

    # Phase 2 — Anchor detection & subject filtering.
    aliases          = build_alias_set(title, entities)
    filtered_triples = filter_triples_by_anchor(triples, aliases)

    # Phase 3 — Relation normalisation.
    entity_lookup    = build_entity_lookup(entities)
    attr_value_pairs = normalise_triples(filtered_triples, entity_lookup)

    # Phase 4 — Deduplication & aggregation.
    attributes = aggregate_attributes(attr_value_pairs)

    # Ensure the person's name is always present as the first attribute.
    if "name" not in attributes:
        attributes["name"] = [title]

    # Phase 5 — Canonical ordering.
    ordered_table = sort_attributes(attributes)

    return {
        "title":                title,
        "num_attributes":       len(ordered_table),
        "num_triples_input":    len(triples),
        "num_triples_filtered": len(filtered_triples),
        "table":                ordered_table,
    }


# ============================================================================
#  OUTPUT WRITERS
# ============================================================================

def write_jsonl(tables: dict[int, dict], output_path: Path) -> None:
    """Write one JSON object per line, one line per record.

    Schema per line::

        {
          "record_id": 0,
          "title": "lenny randle",
          "num_attributes": 5,
          "num_triples_input": 6,
          "num_triples_filtered": 4,
          "table": {
            "name": "lenny randle",
            "birth_date": "february 12, 1949",
            "occupation": ["baseball player", "..."],
            ...
          }
        }

    Single-valued attributes are written as plain strings; multi-valued
    attributes as lists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rid in sorted(tables):
            tbl = tables[rid]

            # Convert ordered list → dict, collapsing single-element lists.
            table_dict: dict[str, str | list[str]] = {}
            for attr, values in tbl["table"]:
                table_dict[attr] = values[0] if len(values) == 1 else values

            obj = {
                "record_id":            rid,
                "title":                tbl["title"],
                "num_attributes":       tbl["num_attributes"],
                "num_triples_input":    tbl["num_triples_input"],
                "num_triples_filtered": tbl["num_triples_filtered"],
                "table":                table_dict,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"  → Wrote {len(tables)} tables to {output_path}")


def write_box(tables: dict[int, dict], output_path: Path) -> None:
    """Write WikiBio .box format — tab-separated indexed tokens, one line per record.

    Compatible with WikiBio evaluation scripts and BLEU-based evaluation.
    Jayanth discussed using seq2seq metrics (BLEU) — this format enables that.

    Format example (one line)::

        name_1:lenny  name_2:randle  birth_date_1:february  birth_date_2:12  ...
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rid in sorted(tables):
            tbl = tables[rid]
            parts: list[str] = []

            for attr, values in tbl["table"]:
                all_tokens: list[str] = []
                for val in values:
                    all_tokens.extend(val.split())

                if not all_tokens:
                    parts.append(f"{attr}:<none>")
                else:
                    for i, tok in enumerate(all_tokens, 1):
                        parts.append(f"{attr}_{i}:{tok}")

            f.write("\t".join(parts) + "\n")

    print(f"  → Wrote {len(tables)} tables in .box format to {output_path}")


def write_markdown(tables: dict[int, dict], output_path: Path) -> None:
    """Write human-readable Markdown tables for manual inspection.

    Useful for the human evaluation the team discussed:
      - Content Relevance: Does the table capture the main facts?
      - Coherence: Are attributes logically grouped?
      - Fluency: Are values readable?
      - Facts: Are values faithful to the source paragraph?

    Each record produces a Markdown table with an Attribute | Value layout.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Generated Tables from Knowledge Graphs\n\n")

        for rid in sorted(tables):
            tbl = tables[rid]
            f.write(f"## {tbl['title']} (record {rid})\n\n")
            f.write(
                f"*Input triples: {tbl['num_triples_input']}"
                f" → Filtered: {tbl['num_triples_filtered']}"
                f" → Attributes: {tbl['num_attributes']}*\n\n"
            )
            f.write("| Attribute | Value |\n")
            f.write("|-----------|-------|\n")

            for attr, values in tbl["table"]:
                display = " ; ".join(values).replace("|", "\\|")
                f.write(f"| {attr} | {display} |\n")

            f.write("\n---\n\n")

    print(f"  → Wrote {len(tables)} tables in Markdown to {output_path}")


# ============================================================================
#  STATISTICS & DIAGNOSTICS
# ============================================================================

def print_statistics(tables: dict[int, dict]) -> None:
    """Print a summary report about the generated tables.

    Reports coverage distribution, filtering effectiveness, and the most
    frequent attributes.  Useful for quick quality assessment.
    """
    total = len(tables)
    if total == 0:
        print("  No tables generated.")
        return

    attr_counts    = [t["num_attributes"]       for t in tables.values()]
    in_counts      = [t["num_triples_input"]    for t in tables.values()]
    filt_counts    = [t["num_triples_filtered"] for t in tables.values()]

    sparse = sum(1 for c in attr_counts if c < 3)
    medium = sum(1 for c in attr_counts if 3 <= c <= 7)
    rich   = sum(1 for c in attr_counts if c > 7)

    attr_freq: defaultdict[str, int] = defaultdict(int)
    for tbl in tables.values():
        for attr, _ in tbl["table"]:
            attr_freq[attr] += 1

    total_in   = sum(in_counts)
    total_filt = sum(filt_counts)

    print(f"\n{'='*60}")
    print(f"  KG → Table Conversion Statistics")
    print(f"{'='*60}")
    print(f"  Total records processed:    {total:,}")
    print(f"  Avg attributes / record:    {sum(attr_counts)/total:.1f}")
    print(f"  Avg triples input:          {total_in/total:.1f}")
    print(f"  Avg triples after filter:   {total_filt/total:.1f}")
    print(f"  Anchor filter retention:    {total_filt/max(total_in,1)*100:.1f}%")
    print()
    print(f"  Coverage distribution:")
    print(f"    Sparse  (< 3 attrs):  {sparse:>6,}  ({sparse/total*100:5.1f}%)")
    print(f"    Medium  (3–7 attrs):  {medium:>6,}  ({medium/total*100:5.1f}%)")
    print(f"    Rich    (> 7 attrs):  {rich:>6,}  ({rich/total*100:5.1f}%)")
    print()
    print(f"  Top 15 most frequent attributes:")
    for attr, cnt in sorted(attr_freq.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * int(cnt / total * 40)
        print(f"    {attr:30s}  {cnt:>6,} ({cnt/total*100:5.1f}%)  {bar}")
    print(f"{'='*60}\n")


# ============================================================================
#  CLI ENTRY POINT
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 6 — Convert knowledge-graph triples to structured tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", type=Path, required=True,
        help='Path to processed triples JSON '
             '(e.g. "Post mid sub files _ rough/processed_triples_44k_refined.json")',
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="Output path (.jsonl / .box / .md)",
    )
    ap.add_argument(
        "--output-format", choices=["jsonl", "box", "markdown"], default="jsonl",
        help="Output format (default: jsonl)",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N records (0 = all).  Useful for testing.",
    )
    ap.add_argument(
        "--no-stats", action="store_true",
        help="Suppress the summary statistics printout.",
    )
    args = ap.parse_args()

    # --- Validate input ---
    if not args.input.exists():
        sys.exit(f"ERROR: Input file not found: {args.input}")

    # --- Phase 1: Load & group ---
    print(f"Loading {args.input} …")
    records = load_and_group(args.input)
    print(f"  Found {len(records):,} unique records")

    # Optional limit for quick testing.
    if args.limit > 0:
        limited = sorted(records.keys())[:args.limit]
        records = {r: records[r] for r in limited}
        print(f"  Limited to first {args.limit} records")

    # --- Phases 2–5: Build tables ---
    print("Building tables …")
    tables: dict[int, dict] = {}
    for rid in sorted(records):
        tables[rid] = build_table(records[rid])

    # --- Statistics ---
    if not args.no_stats:
        print_statistics(tables)

    # --- Write output ---
    if args.output_format == "box":
        write_box(tables, args.output)
    elif args.output_format == "markdown":
        write_markdown(tables, args.output)
    else:
        write_jsonl(tables, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
