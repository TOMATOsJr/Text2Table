import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def resolve_identities(input_file, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Nomic Embeddings on {device}...")
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True).to(device)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resolved_records = []
    
    for record in data:
        title = record['title']
        triples = record['triples']
        
        # Collect all entities in this biography
        entities = set()
        for t in triples:
            entities.add(t['subject'])
            entities.add(t['object'])
        
        if not entities:
            resolved_records.append(record)
            continue
            
        entity_list = list(entities)
        
        # Get embeddings for Title and all observed entities
        # Use prefix for Nomic search tasks (classification-like)
        title_emb = model.encode([f"search_query: {title}"], convert_to_tensor=True)
        entity_embs = model.encode([f"search_document: {e}" for e in entity_list], convert_to_tensor=True)
        
        # Calculate Cosine Similarities
        cos_sim = torch.nn.functional.cosine_similarity(title_emb, entity_embs)
        
        # Mapping for identity resolution
        identity_map = {}
        identity_log = []
        
        for i, ent in enumerate(entity_list):
            sim = cos_sim[i].item()
            # 0.70 threshold for canonicalization
            if sim > 0.70 or ent.lower() in title.lower() or title.lower() in ent.lower():
                identity_map[ent] = title
                identity_log.append(f"MERGE: '{ent}' -> '{title}' (Sim: {sim:.2f})")
            else:
                identity_map[ent] = ent # Keep as is
                identity_log.append(f"KEEP : '{ent}' (Sim: {sim:.2f})")

        # Update triples with resolved identities
        resolved_triples = []
        for t in triples:
            resolved_t = t.copy()
            resolved_t['subject_original'] = t['subject']
            resolved_t['object_original'] = t['object']
            resolved_t['subject'] = identity_map.get(t['subject'], t['subject'])
            resolved_t['object'] = identity_map.get(t['object'], t['object'])
            resolved_triples.append(resolved_t)
            
        resolved_records.append({
            "record_id": record['record_id'],
            "title": title,
            "sentence": record['sentence'],
            "identity_resolution_log": identity_log,
            "triples": resolved_triples
        })

    print(f"Saving Identity-resolved KG to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(resolved_records, f, indent=2)

if __name__ == "__main__":
    resolve_identities('sample_data/refined_triples.json', 'sample_data/identity_resolved_kg.json')
