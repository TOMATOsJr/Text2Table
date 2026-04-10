import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def create_embeddings(input_file: str, output_file: str, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
    """
    Reads extracted triples from input_file, generates embeddings, 
    and saves them alongside provenance metadata.
    """
    print(f"Loading researcher-grade model: {model_name}...")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)
    
    print(f"Reading {input_file}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # Try loading as a single JSON object first (e.g. a list or a single dict)
    try:
        records = json.loads(content)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        # Fallback: Handle concatenated JSON objects (not in a single list or JSONL)
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            # Skip potential whitespace between objects
            content = content.lstrip()
            if not content:
                break
            try:
                record, pos = decoder.raw_decode(content)
                records.append(record)
                content = content[pos:].lstrip()
                # Reset relative pos since we sliced content
                pos = 0
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at position {pos}: {e}")
                break

    print(f"Found {len(records)} records. Extracting entities and relations...")
    
    # Collect all unique entities and relations to embed them only once
    unique_entities = set()
    unique_relations = set()
    
    for record in records:
        # Crucial: Always embed the Anchor Node (title)
        if record.get('title'):
            unique_entities.add(record['title'].strip().lower())
            
        for triple in record.get('filtered_triples', []):
            unique_entities.add(triple['subject'].strip().lower())
            unique_entities.add(triple['object'].strip().lower())
            unique_relations.add(triple['relation'].strip().lower())
    
    entity_list = sorted(list(unique_entities))
    relation_list = sorted(list(unique_relations))
    
    print(f"Encoding {len(entity_list)} unique entities and {len(relation_list)} unique relations...")
    
    # Generate embeddings
    entity_embeddings = model.encode(entity_list, show_progress_bar=True, device=device)
    relation_embeddings = model.encode(relation_list, show_progress_bar=True, device=device)
    
    # Create lookup dictionaries for similarity checks
    entity_lookup = {text: emb.tolist() for text, emb in zip(entity_list, entity_embeddings)}
    relation_lookup = {text: emb.tolist() for text, emb in zip(relation_list, relation_embeddings)}
    
    # Prepare the refined data structure with metadata-based traceability
    # We will NOT concatenate IDs to vectors here, but store them in the JSON structure.
    refined_output = {
        "metadata": {
            "model": model_name,
            "device": device,
            "entity_dim": len(entity_embeddings[0]),
            "relation_dim": len(relation_embeddings[0])
        },
        "embeddings": {
            "entities": entity_lookup,
            "relations": relation_lookup
        },
        "records": []
    }
    
    for record in records:
        refined_record = {
            "record_id": record.get("record_id"),
            "title": record.get("title"),
            "triples": []
        }
        for triple in record.get("filtered_triples", []):
            # Each triple is now enriched with text-to-embedding-lookup capability
            refined_triple = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "provenance": {
                    "record_id": record.get("record_id"),
                    "sentence_id": record.get("sentence_id"),
                    "sentence": record.get("sentence")
                }
            }
            refined_record["triples"].append(refined_triple)
        refined_output["records"].append(refined_record)
        
    print(f"Saving refined results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(refined_output, f, indent=2)
    
    print("Embedding generation complete!")

if __name__ == "__main__":
    create_embeddings('output.json', 'refined_embeddings.json')
