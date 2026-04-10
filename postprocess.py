"""Phase 4 – Post-process REBEL JSONL into per-record attribute tables.

Reads the sentence-level JSONL produced by relation_extraction.py, normalises
REBEL relation names to WikiBio-style attribute keys, deduplicates values, and
writes one of two output formats:

  --output-format records   (default)  One JSON object per record with merged attributes.
  --output-format box       WikiBio .box-compatible format (tab-separated key_N:value).

Usage:
    python postprocess.py --input outputs/triples.jsonl --output outputs/records.jsonl
    python postprocess.py --input outputs/triples.jsonl --output outputs/pred.box --output-format box
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Relation → WikiBio attribute mapping
# ---------------------------------------------------------------------------

# Maps REBEL / Wikidata relation names to WikiBio .box attribute keys.
# Relations not listed here are normalised automatically via _auto_normalise().
RELATION_TO_ATTRIBUTE: dict[str, str] = {
    # Birth / death
    "date of birth": "birth_date",
    "place of birth": "birth_place",
    "date of death": "death_date",
    "place of death": "death_place",
    "cause of death": "death_cause",
    "inception": "birth_date",
    # Personal
    "sex or gender": "gender",
    "country of citizenship": "nationality",
    "nationality": "nationality",
    "religion": "religion",
    "residence": "residence",
    "spouse": "spouse",
    "child": "children",
    "father": "parents",
    "mother": "parents",
    "sibling": "relations",
    "relative": "relations",
    # Education / career
    "educated at": "alma_mater",
    "alma mater": "alma_mater",
    "academic degree": "education",
    "occupation": "occupation",
    "employer": "employer",
    "position held": "office",
    "field of work": "known_for",
    # Sports
    "member of sports team": "clubs",
    "sport": "sport",
    "position played on team": "position",
    "coach of sports team": "clubs",
    "country for sport": "nationality",
    # Awards / recognition
    "award received": "awards",
    "nominated for": "awards",
    "member of": "member_of",
    "military rank": "military_rank",
    "conflict": "conflict",
    # Media / arts
    "genre": "genre",
    "instrument": "instrument",
    "record label": "label",
    "performer": "associated_acts",
    "notable work": "notable_work",
    "author": "notable_work",
    "publisher": "publisher",
    "language of work or name": "language",
    # Politics
    "member of political party": "party",
    # Geography / admin
    "country": "country",
    "located in the administrative territorial entity": "location",
    "contains administrative territorial entity": "subdivisions",
    "place of burial": "resting_place",
    # Temporal
    "point in time": "year",
    # Structural (less useful for biography tables, but kept)
    "instance of": "instance_of",
    "part of": "part_of",
    "has part": "has_part",
    "subclass of": "subclass_of",
    "different from": "different_from",
    "follows": "follows",
    "followed by": "followed_by",
    "parent organization": "parent_organization",
    "subsidiary": "subsidiary",
    "applies to jurisdiction": "jurisdiction",
    "legislated by": "legislated_by",
    "participant of": "known_for",
    "participant in": "known_for",
}


def _auto_normalise(relation: str) -> str:
    """Convert an unmapped relation string to a snake_case attribute key."""
    key = relation.strip().lower()
    key = re.sub(r"[^a-z0-9\s]", "", key)
    key = re.sub(r"\s+", "_", key).strip("_")
    return key or "other"


def normalise_relation(relation: str) -> str:
    """Map a REBEL relation to a WikiBio-style attribute key."""
    rel = relation.strip().lower()
    if rel in RELATION_TO_ATTRIBUTE:
        return RELATION_TO_ATTRIBUTE[rel]
    
    # 3. Handle 'born', 'died' variants (from zip core)
    if "born" in rel:
        if "place" in rel or "in" in rel: return "birth_place"
        return "birth_date"
    if "died" in rel or "death" in rel:
        if "place" in rel or "in" in rel: return "death_place"
        return "death_date"
        
    return _auto_normalise(rel)


# ---------------------------------------------------------------------------
# Record grouping
# ---------------------------------------------------------------------------

def group_by_record(jsonl_path: Path) -> dict[int, dict]:
    """Read sentence-level JSONL and group triples into per-record tables.

    Returns {record_id: {"title": str, "attributes": {attr: [values]}, "sentences": int}}
    """
    records: dict[int, dict] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rid = entry["record_id"]
            if rid not in records:
                records[rid] = {
                    "title": entry.get("title", ""),
                    "attributes": defaultdict(list),
                    "sentences": 0,
                    "total_triples": 0,
                }
            rec = records[rid]
            rec["sentences"] += 1

            for triple in entry.get("filtered_triples", []):
                attr = normalise_relation(triple["relation"])
                subj = triple["subject"].strip()
                obj = triple["object"].strip()

                # For biography tables the object is the cell value.
                val = obj
                # If subject is NOT the person (title), attach it as context.
                # Use token overlap to handle partial name matches
                # (e.g. "shenoff randle" for title "lenny randle").
                title_tokens = set(rec["title"].lower().split())
                subj_tokens = set(subj.lower().split())
                is_person = bool(title_tokens & subj_tokens)
                if not is_person:
                    val = f"{subj}: {obj}"

                # Dedup: skip exact duplicate values for same attribute.
                if val not in rec["attributes"][attr]:
                    rec["attributes"][attr].append(val)
                rec["total_triples"] += 1

    return records


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def write_records_jsonl(records: dict[int, dict], output_path: Path) -> None:
    """Write per-record JSON, one line per record."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rid in sorted(records):
            rec = records[rid]
            obj = {
                "record_id": rid,
                "title": rec["title"],
                "sentences": rec["sentences"],
                "total_triples": rec["total_triples"],
                "attributes": dict(rec["attributes"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


def write_box_format(records: dict[int, dict], output_path: Path) -> None:
    """Write WikiBio .box-compatible format (one line per record)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rid in sorted(records):
            rec = records[rid]
            parts: list[str] = []
            # Add article_title from record title.
            title_tokens = rec["title"].split()
            for i, tok in enumerate(title_tokens, 1):
                parts.append(f"article_title_{i}:{tok}")

            for attr, values in sorted(rec["attributes"].items()):
                # Flatten multi-valued attributes into indexed tokens like WikiBio.
                all_tokens: list[str] = []
                for val in values:
                    all_tokens.extend(val.split())
                if not all_tokens:
                    parts.append(f"{attr}:<none>")
                else:
                    for i, tok in enumerate(all_tokens, 1):
                        parts.append(f"{attr}_{i}:{tok}")
            f.write("\t".join(parts) + "\n")
    print(f"Wrote {len(records)} records in .box format to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Post-process REBEL JSONL into per-record tables."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to sentence-level JSONL from relation_extraction.py",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path (.jsonl for records, or .box for WikiBio format)",
    )
    parser.add_argument(
        "--output-format", type=str, choices=["records", "box"], default="records",
        help="Output format: per-record JSONL or WikiBio .box",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    records = group_by_record(args.input)

    # Print quick summary.
    total_attrs = sum(len(r["attributes"]) for r in records.values())
    total_vals = sum(
        sum(len(v) for v in r["attributes"].values()) for r in records.values()
    )
    print(f"Grouped {total_vals} values across {total_attrs} attributes in {len(records)} records")

    if args.output_format == "box":
        write_box_format(records, args.output)
    else:
        write_records_jsonl(records, args.output)


if __name__ == "__main__":
    main()
