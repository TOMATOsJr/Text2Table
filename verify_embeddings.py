import json
import numpy as np

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def verify_results(file_path):
    print(f"Loading {file_path} for verification...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities = data['embeddings']['entities']
    
    # Test Case 1: Lenny Randle (Record 0)
    title = "lenny randle"
    extracted = "shenoff randle"
    
    if title in entities and extracted in entities:
        sim = cosine_similarity(entities[title], entities[extracted])
        print(f"Similarity ('{title}', '{extracted}'): {sim:.4f}")
        if sim > 0.8:
            print("Status: SUCCESS (High similarity found for name variants)")
    else:
        print(f"Error: One of the entities not found in embeddings. Available: {list(entities.keys())[:5]}")

    # Test Case 2: Relationship check (Self-Similarity should be 1.0)
    if title in entities:
        sim_self = cosine_similarity(entities[title], entities[title])
        print(f"Self-Similarity ('{title}'): {sim_self:.4f}")

    # Provenance Check
    first_record = data['records'][0]
    first_triple = first_record['triples'][0]
    print(f"\nProvenance Check (Record {first_record['record_id']}):")
    print(f"Triple: {first_triple['subject']} --{first_triple['relation']}--> {first_triple['object']}")
    print(f"Source Sentence: {first_triple['provenance']['sentence'][:50]}...")

if __name__ == "__main__":
    verify_results('refined_embeddings.json')
