import json
from typing import List, Dict, Any

def get_identity_score(triple: Dict[str, Any], anchor_title: str) -> bool:
    """
    Returns True if the subject of the triple is statistically likely to be the Anchor Node.
    Uses Hybrid logic: Neural + Hyperbolic + SDP.
    """
    # 1. Neural Score (from earlier check, we know 0.66 is a good cutoff for Nomic)
    # Since we don't store Sim in the JSON (only Radius/SDP), we'll use Radius + SDP
    
    # 2. Geometric Radius (Radius < 4.85 is 'Root' in our log-dampened space)
    radius = triple.get('subject_radius', 9.9)
    is_geometrically_close = radius < 4.9
    
    # 3. Syntactic Grounding
    path = triple.get('sdp_path', '').lower()
    is_syntactically_grounded = any(word in path for word in ['born', 'is', 'was', 'member'])
    
    # Decision Matrix:
    # If it's the root of a 'born' or 'is' sentence, it's almost certainly the subject.
    if is_syntactically_grounded and radius < 5.0:
        return True
        
    # If it's mathematically at the 'center' of the biography.
    if is_geometrically_close:
        return True
        
    return False

def finalize_kg(input_file: str, output_file: str):
    print(f"Loading grounded data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    refined_records = []
    
    print("Performing Hybrid Identity Merging...")
    for record in data['records']:
        title = record['title']
        clean_triples = []
        
        for triple in record['triples']:
            # Check Subject Identity
            if get_identity_score(triple, title):
                # MERGE: Replace variant with canonical Title
                triple['subject'] = title
                
            # Check Object Identity (for cases where the subject is 'mother of X')
            # For now, we focus on Subject-Centric pruning
            
            # PRUNING LOGIC:
            # If the subject is NOT the anchor AND not grounded, it's peripheral.
            if triple['subject'] == title:
                clean_triples.append(triple)
            else:
                # Keep it if it's a high-confidence relation even if not 'identity'
                # (e.g. Spouse relations) - for now, we follow the 'Strict Bio' rule.
                pass
                
        record['triples'] = clean_triples
        refined_records.append(record)
        
    print(f"Saving final KG to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("Identity Filtering complete!")

if __name__ == "__main__":
    finalize_kg('grounded_triples.json', 'final_refined_kg.json')
