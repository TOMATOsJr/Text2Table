import re
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

class CanonicalKnowledgeGraph:
    """
    A globally consolidated Knowledge Graph built from canonicalized triples.
    
    Features:
    - Identity Resolution: Merges duplicate nodes using canonical IDs.
    - Confidence Filtering: Drops low-confidence relation mappings.
    - Formal Export: Serializes to Turtle (TTL).
    """
    def __init__(self, namespace: str = "http://example.org/bio/"):
        self.namespace = namespace
        self.triples: set[Tuple[str, str, str]] = set()
        self.nodes: set[str] = set()
        self.stats = {"triples": 0, "nodes": 0}

    def add_from_json(self, data: List[Dict], threshold: float = 0.1):
        """Aggregate triples from a list of record-based JSON objects."""
        added = 0
        for record in data:
            for t in record.get("triples", []):
                # prioritize canonical fields from Stage 5
                subj = t.get("canonical_subject") or t.get("subject")
                rel = t.get("canonical_relation") or t.get("relation")
                obj = t.get("canonical_object") or t.get("object")
                conf = t.get("canonical_confidence", 1.0)

                if rel == "__unmapped__" or not rel:
                    continue
                if conf < threshold:
                    continue

                # Clean strings
                subj, rel, obj = str(subj).strip(), str(rel).strip(), str(obj).strip()
                if not (subj and rel and obj):
                    continue

                self.triples.add((subj, rel, obj))
                self.nodes.add(subj)
                self.nodes.add(obj)
                added += 1
        
        self.stats["triples"] = len(self.triples)
        self.stats["nodes"] = len(self.nodes)
        print(f"[KG] Built graph with {self.stats['triples']:,} unique edges and {self.stats['nodes']:,} nodes.")

    def _to_uri(self, text: str) -> str:
        """Convert a string label to a safe URI fragment."""
        # Simple URI safety: alphanumeric and underscores
        clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', text)
        clean = re.sub(r'_+', '_', clean).strip('_')
        return f"<{self.namespace}{clean}>"

    def to_turtle(self, output_path: str):
        """Export the consolidated graph to Turtle (TTL) format."""
        print(f"[KG] Serializing to {output_path}…")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"@prefix : <{self.namespace}> .\n")
            f.write(f"@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n")
            
            for s, r, o in sorted(self.triples):
                s_uri = self._to_uri(s)
                r_uri = self._to_uri(r)
                # Heuristic: if object is short and looks like a label, it's a URI node.
                # Else it's a literal.
                if len(o) > 40 or re.search(r'\d', o) or "," in o:
                    o_val = f'"{o}"'
                else:
                    o_val = self._to_uri(o)
                
                f.write(f"{s_uri} {r_uri} {o_val} .\n")
        print("[KG] Export complete.")


    def to_json(self, output_path: str):
        """Export the graph in a 'Nodes and Edges' JSON format."""
        print(f"[KG] Serializing to JSON {output_path}…")
        graph_data = {
            "nodes": [{"id": node, "label": node} for node in sorted(self.nodes)],
            "edges": [{"source": s, "target": o, "relation": r} for s, r, o in sorted(self.triples)]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        print("[KG] JSON Export complete.")


# Legacy compatibility functions
def build_knowledge_graph(triples: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, List[str]]]:
    """Legacy dictionary-based builder."""
    kg = defaultdict(lambda: defaultdict(list))
    for subj, rel, obj in triples:
        if obj not in kg[subj][rel]:
            kg[subj][rel].append(obj)
    return {s: dict(r) for s, r in kg.items()}

def print_knowledge_graph(kg: Dict[str, Dict[str, List[str]]]) -> None:
    for subject, relations in kg.items():
        print(f"\n{subject}")
        for relation, objects in relations.items():
            for obj in objects:
                print(f"  ├── {relation} -> {obj}")