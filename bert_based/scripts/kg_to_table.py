#!/usr/bin/env python3
"""Convert knowledge graph JSON into 2-column per-person table blocks.

Input format:
{
  "person title": {
	"relation_a": ["value1", "value2"],
	"relation_b": ["value3"]
  },
  ...
}

Output CSV format (2 columns):
Key,Value
Title,person title
relation_a,value1
relation_a,value2
relation_b,value3

[blank line]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert knowledge graph JSON to 2-column tables.")
	parser.add_argument(
		"--input_json",
		type=Path,
		default=Path("../outputs/knowledge_graph_predictions_canonical.json"),
		help="Path to knowledge graph JSON file.",
	)
	parser.add_argument(
		"--output_csv",
		type=Path,
		default=Path("../outputs/knowledge_graph_predictions_canonical_table.csv"),
		help="Path to output 2-column CSV file.",
	)
	return parser.parse_args()


def load_kg(path: Path) -> Dict[str, Dict[str, List[str]]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError("Top-level JSON must be a dictionary: title -> relations.")
	return data


def write_tables(kg: Dict[str, Dict[str, List[str]]], output_csv: Path) -> Dict[str, int]:
	output_csv.parent.mkdir(parents=True, exist_ok=True)

	edge_count = 0
	node_count = 0

	with output_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["Key", "Value"])

		titles = list(kg.keys())
		for i, title in enumerate(tqdm(titles, total=len(titles), desc="kg->table", unit="person")):
			relations = kg.get(title, {})
			writer.writerow(["Title", title])

			if isinstance(relations, dict):
				for relation, values in relations.items():
					if isinstance(values, list):
						for value in values:
							writer.writerow([relation, value])
							edge_count += 1
							node_count += 1
					else:
						writer.writerow([relation, values])
						edge_count += 1
						node_count += 1

			# Empty line after each person's table block.
			writer.writerow([])

	return {"titles": len(kg), "edges": edge_count, "nodes": node_count}


def main() -> None:
	args = parse_args()

	if not args.input_json.exists():
		raise FileNotFoundError(f"Missing input file: {args.input_json}")

	kg = load_kg(args.input_json)
	stats = write_tables(kg, args.output_csv)

	print(f"Saved table CSV: {args.output_csv}")
	print(f"People processed: {stats['titles']}")
	print(f"Edges written: {stats['edges']}")
	print(f"Nodes counted: {stats['nodes']}")
	print("Note: nodes are counted as value/object nodes only.")


if __name__ == "__main__":
	main()
