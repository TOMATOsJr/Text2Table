"""Phase 6 - Knowledge Graph to Grouped Text Conversion.

This is a grouped-text variant of kg_to_table.py.

It converts extracted triples from the hybrid REBEL + GLiNER pipeline into a
grouped text format where each biography becomes a small block:

    Title: Person x1
    birth_place: New York,
    occupation: writer

    Title: Person x2
    birth_place: London,
    occupation: actor

The relation normalization step can be switched on or off from the CLI.

When normalization is enabled, the script:
  - splits concatenated relations
  - maps relations to canonical WikiBio-style attributes
  - disambiguates birth/death date vs place using entity labels
  - drops structural/noisy relations

When normalization is disabled, the script keeps the raw relation text,
sanitized into a snake_case attribute name.

Usage:
  python kg_to_table_csv.py \
      --input "Post mid sub files _ rough/processed_triples_44k_refined.json" \
      --output outputs/tables.txt

  python kg_to_table_csv.py \
      --input "Post mid sub files _ rough/processed_triples_44k_refined.json" \
      --output outputs/tables_raw.txt \
      --no-normalize-relations
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from postprocess import RELATION_TO_ATTRIBUTE, normalise_relation, _auto_normalise
except ImportError:
    _project_root = str(Path(__file__).resolve().parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from postprocess import RELATION_TO_ATTRIBUTE, normalise_relation, _auto_normalise


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

SINGLE_VALUED_ATTRS: set[str] = {
    "name", "fullname", "birth_name",
    "birth_date", "birth_place",
    "death_date", "death_place", "death_cause",
    "gender", "nationality", "religion",
}

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

LOCATION_LABELS: set[str] = {
    "GPE", "LOC", "FAC",
    "gpe", "loc", "fac",
    "location", "place", "city", "country",
}

DATE_LABELS: set[str] = {
    "DATE", "TIME",
    "date", "time",
    "CARDINAL", "ORDINAL",
}

MIN_ANCHOR_TOKEN_LEN = 3

DROP_ATTRIBUTES: set[str] = {
    "instance_of", "subclass_of", "different_from",
    "part_of", "has_part",
    "follows", "followed_by",
    "applies_to_jurisdiction", "legislated_by",
    "contains_administrative_territorial_entity",
    "located_in_the_administrative_territorial_entity",
}


def load_and_group(input_path: Path) -> dict[int, dict]:
    """Load the sentence-level JSON array and group it by record_id."""
    with input_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    records: dict[int, dict] = {}
    skipped = 0

    for entry in raw_data:
        title = entry.get("title", "").strip()
        if not title:
            skipped += 1
            continue

        record_id = entry["record_id"]
        if record_id not in records:
            records[record_id] = {
                "title": title,
                "triples": [],
                "entities": [],
                "sentences": [],
            }

        record = records[record_id]
        record["triples"].extend(entry.get("triples", []))
        record["entities"].extend(entry.get("entities", []))

        sentence = entry.get("sentence", "")
        if sentence and sentence not in record["sentences"]:
            record["sentences"].append(sentence)

    if skipped:
        print(f"  Skipped {skipped:,} entries with empty titles")

    return records


def build_alias_set(title: str, entities: list[dict]) -> set[str]:
    aliases: set[str] = set()

    title_lower = title.strip().lower()
    if title_lower:
        aliases.add(title_lower)

    for entity in entities:
        if entity.get("label", "").upper() == "PERSON":
            entity_text = entity.get("text", "").strip().lower()
            if entity_text:
                aliases.add(entity_text)

    token_aliases: set[str] = set()
    for alias in list(aliases):
        for token in alias.split():
            if len(token) >= 4:
                token_aliases.add(token)
    aliases.update(token_aliases)

    return aliases


def subject_matches_anchor(subject: str, aliases: set[str]) -> bool:
    subject_lower = subject.strip().lower()

    if subject_lower in aliases:
        return True

    for token in subject_lower.split():
        if len(token) >= MIN_ANCHOR_TOKEN_LEN and token in aliases:
            return True

    return False


def filter_triples_by_anchor(triples: list[dict], aliases: set[str]) -> list[dict]:
    return [triple for triple in triples if subject_matches_anchor(triple["subject"], aliases)]


def split_concatenated_relation(relation: str) -> list[str]:
    rel = relation.strip().lower()

    if rel in SEED_RELATIONS or rel in RELATION_TO_ATTRIBUTE:
        return [rel]

    sorted_relations = sorted(SEED_RELATIONS, key=len, reverse=True)
    found: list[str] = []
    remaining = rel

    while remaining:
        matched = False
        for known_relation in sorted_relations:
            if remaining.startswith(known_relation):
                found.append(known_relation)
                remaining = remaining[len(known_relation):].strip()
                matched = True
                break
        if not matched:
            break

    if len(found) >= 2 and remaining.strip() == "":
        return found

    return [rel]


def disambiguate_date_place(attribute: str, value: str, entity_lookup: dict[str, str]) -> str:
    date_attrs = {"birth_date", "death_date"}
    place_attrs = {"birth_place", "death_place"}

    if attribute not in date_attrs and attribute not in place_attrs:
        return attribute

    value_lower = value.strip().lower()
    for entity_text, entity_label in entity_lookup.items():
        if entity_text in value_lower or value_lower in entity_text:
            if attribute in date_attrs and entity_label in LOCATION_LABELS:
                return attribute.replace("date", "place")
            if attribute in place_attrs and entity_label in DATE_LABELS:
                return attribute.replace("place", "date")

    return attribute


def build_entity_lookup(entities: list[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entity in entities:
        text = entity.get("text", "").strip().lower()
        label = entity.get("label", "").upper()
        if text and label:
            lookup[text] = label
    return lookup


def normalize_triples(
    triples: list[dict],
    entity_lookup: dict[str, str],
    normalize_relations: bool,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for triple in triples:
        relation = triple["relation"].strip()
        value = triple["object"].strip()

        if not value:
            continue

        if normalize_relations:
            relations = split_concatenated_relation(relation)
        else:
            relations = [relation]

        for rel in relations:
            if normalize_relations:
                attribute = normalise_relation(rel)
                attribute = disambiguate_date_place(attribute, value, entity_lookup)
                if attribute in DROP_ATTRIBUTES:
                    continue
            else:
                attribute = _auto_normalise(rel)

            pairs.append((attribute, value))

    return pairs


def deduplicate_values(values: list[str]) -> list[str]:
    if not values:
        return values

    seen_lower: set[str] = set()
    unique: list[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        value_lower = value.lower()
        if value_lower not in seen_lower:
            seen_lower.add(value_lower)
            unique.append(value)

    if len(unique) <= 1:
        return unique

    sorted_by_length = sorted(unique, key=len, reverse=True)
    kept: list[str] = []
    for value in sorted_by_length:
        value_lower = value.lower()
        absorbed = any(
            value_lower in accepted.lower() and value_lower != accepted.lower()
            for accepted in kept
        )
        if not absorbed:
            kept.append(value)

    return kept


def pick_best_single_value(values: list[str]) -> str:
    if not values:
        return ""
    return max(values, key=len)


def aggregate_attributes(attr_value_pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for attribute, value in attr_value_pairs:
        grouped[attribute].append(value)

    result: dict[str, list[str]] = {}
    for attribute, values in grouped.items():
        deduped = deduplicate_values(values)
        if not deduped:
            continue

        if attribute in SINGLE_VALUED_ATTRS:
            result[attribute] = [pick_best_single_value(deduped)]
        else:
            result[attribute] = deduped

    return result


def sort_attributes(attributes: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    ordered: list[tuple[str, list[str]]] = []
    remaining = set(attributes.keys())

    for attribute in ATTRIBUTE_ORDER:
        if attribute in attributes:
            ordered.append((attribute, attributes[attribute]))
            remaining.discard(attribute)

    for attribute in sorted(remaining):
        ordered.append((attribute, attributes[attribute]))

    return ordered


def build_table(record: dict, normalize_relations: bool) -> dict[str, Any]:
    title = record["title"]
    triples = record["triples"]
    entities = record["entities"]

    aliases = build_alias_set(title, entities)
    filtered_triples = filter_triples_by_anchor(triples, aliases)

    entity_lookup = build_entity_lookup(entities)
    attr_value_pairs = normalize_triples(filtered_triples, entity_lookup, normalize_relations)
    attributes = aggregate_attributes(attr_value_pairs)

    ordered_table = sort_attributes(attributes)

    return {
        "title": title,
        "num_attributes": len(ordered_table),
        "num_triples_input": len(triples),
        "num_triples_filtered": len(filtered_triples),
        "table": ordered_table,
    }


def write_grouped_text(tables: dict[int, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, record_id in enumerate(sorted(tables)):
            table = tables[record_id]
            handle.write(f"Title: {table['title']}\n")

            for attribute, values in table["table"]:
                for value in values:
                    handle.write(f"{attribute}: {value}\n")

            if index < len(tables) - 1:
                handle.write("\n")

    print(f"  -> Wrote {len(tables)} grouped records to {output_path}")


def print_statistics(tables: dict[int, dict]) -> None:
    total = len(tables)
    if total == 0:
        print("  No tables generated.")
        return

    attr_counts = [table["num_attributes"] for table in tables.values()]
    in_counts = [table["num_triples_input"] for table in tables.values()]
    filt_counts = [table["num_triples_filtered"] for table in tables.values()]

    total_in = sum(in_counts)
    total_filt = sum(filt_counts)

    print(f"\n{'='*60}")
    print("  KG -> Grouped Text Conversion Statistics")
    print(f"{'='*60}")
    print(f"  Total records processed:    {total:,}")
    print(f"  Avg attributes / record:    {sum(attr_counts) / total:.1f}")
    print(f"  Avg triples input:          {total_in / total:.1f}")
    print(f"  Avg triples after filter:   {total_filt / total:.1f}")
    print(f"  Anchor filter retention:    {total_filt / max(total_in, 1) * 100:.1f}%")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 - Convert knowledge-graph triples to grouped text blocks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help='Path to processed triples JSON (for example, "Post mid sub files _ rough/processed_triples_44k_refined.json")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the grouped text file",
    )
    parser.add_argument(
        "--normalize-relations",
        dest="normalize_relations",
        action="store_true",
        default=True,
        help="Enable relation normalization and date/place disambiguation (default)",
    )
    parser.add_argument(
        "--no-normalize-relations",
        dest="normalize_relations",
        action="store_false",
        help="Disable relation normalization and keep raw relation names as attributes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N records (0 = all)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Suppress the summary statistics printout",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: Input file not found: {args.input}")

    print(f"Loading {args.input} ...")
    records = load_and_group(args.input)
    print(f"  Found {len(records):,} unique records")

    if args.limit > 0:
        limited_ids = sorted(records.keys())[:args.limit]
        records = {record_id: records[record_id] for record_id in limited_ids}
        print(f"  Limited to first {args.limit} records")

    print("Building tables ...")
    tables: dict[int, dict] = {}
    for record_id in sorted(records):
        tables[record_id] = build_table(records[record_id], args.normalize_relations)

    if not args.no_stats:
        mode = "on" if args.normalize_relations else "off"
        print(f"  Relation normalization: {mode}")
        print_statistics(tables)

    write_grouped_text(tables, args.output)
    print("Done.")


if __name__ == "__main__":
    main()