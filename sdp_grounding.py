import json
import spacy
import networkx as nx
from typing import List, Dict, Any, Tuple
import argparse
from pathlib import Path

# Load the small English model for speed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_sdp_path(sentence: str, source_text: str, target_text: str) -> List[str]:
    """
    Finds the shortest dependency path between source and target entities in a sentence.
    """
    doc = nlp(sentence)
    
    source_tokens = []
    target_tokens = []
    
    source_text = source_text.lower()
    target_text = target_text.lower()
    
    for token in doc:
        t_text = token.text.lower()
        if t_text in source_text.split():
            source_tokens.append(token.i)
        if t_text in target_text.split():
            target_tokens.append(token.i)
            
    if not source_tokens or not target_tokens:
        return []

    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i))
            
    graph = nx.Graph(edges)
    
    best_path = []
    min_len = float('inf')
    
    for s in source_tokens:
        for t in target_tokens:
            try:
                path = nx.shortest_path(graph, source=s, target=t)
                if len(path) < min_len:
                    min_len = len(path)
                    best_path = path
            except nx.NetworkXNoPath:
                continue
                
    return [doc[i].text for i in best_path]

def process_sdp(input_file: str, output_file: str):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict-wrapped records
    records = data if isinstance(data, list) else data.get('records', [])
    
    print("Grounding triples in syntactic dependency paths...")
    for record in records:
        for triple in record.get('triples', []):
            sentence = record.get('provenance_sentence', '') # Use the record's provenance
            if not sentence: continue
                
            sdp_tokens = get_sdp_path(sentence, triple['subject'], triple['object'])
            triple['sdp_path'] = " -> ".join(sdp_tokens)
            
            if not sdp_tokens:
                triple['sdp_status'] = "DISCONNECTED"
            elif len(sdp_tokens) <= 2:
                triple['sdp_status'] = "DIRECT"
            else:
                triple['sdp_status'] = "GROUNDED"
                
    print(f"Saving grounded data to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print("SDP Grounding complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syntactic Grounding of Knowledge Graph Triples.")
    parser.add_argument("--input", type=str, required=True, help="Input KG JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output grounded JSON file")
    
    args = parser.parse_args()
    process_sdp(args.input, args.output)
