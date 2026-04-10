import json
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def diagnose():
    with open('grounded_triples.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    r0 = data['records'][0]
    title = r0['title'].lower()
    entities = data['embeddings']['entities']
    title_vec = entities[title]
    
    print(f"ANCHOR SUBJECT: {title.upper()}")
    print("-" * 50)
    
    targets = ["shenoff randle", "major league", "washington senators", "1970 major league baseball draft"]
    
    for target in targets:
        if target not in entities:
            print(f"Entity '{target}' not found in current 20 records.")
            continue
            
        t_vec = entities[target]
        sim = cosine_similarity(title_vec, t_vec)
        
        # Find the triple containing this subject and check its radius
        radius = "N/A"
        sdp = "N/A"
        for t in r0['triples']:
            if t['subject'].lower() == target:
                radius = t['subject_radius']
                sdp = t['sdp_path']
                break
        
        print(f"ENTITY: {target:35}")
        print(f"  > Neural Similarity: {sim:.4f}")
        print(f"  > Hyperbolic Radius: {radius}")
        print(f"  > SDP Path:         {sdp}")
        print("-" * 30)

if __name__ == "__main__":
    diagnose()
