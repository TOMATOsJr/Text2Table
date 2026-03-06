from collections import defaultdict
from typing import List, Tuple, Dict, Any

def build_knowledge_graph(triples: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a simple knowledge graph from (subject, relation, object) triples.

    Output format:
    {
        subject: {
            relation1: [object1, object2, ...],
            relation2: [object3, ...]
        }
    }
    """

    kg = defaultdict(lambda: defaultdict(list))

    for subj, rel, obj in triples:
        # basic cleaning
        subj = subj.strip()
        rel = rel.strip()
        obj = obj.strip()

        if not subj or not rel or not obj:
            continue

        # avoid duplicate edges
        if obj not in kg[subj][rel]:
            kg[subj][rel].append(obj)

    # convert nested defaultdicts to normal dicts
    kg = {s: dict(r) for s, r in kg.items()}
    return kg

def print_knowledge_graph(kg: Dict[str, Dict[str, List[str]]]) -> None:
    for subject, relations in kg.items():
        print(f"\n{subject}")
        for relation, objects in relations.items():
            for obj in objects:
                print(f"  ├── {relation} -> {obj}")