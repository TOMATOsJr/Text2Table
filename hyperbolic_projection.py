import json
import numpy as np
from typing import List, Dict, Any

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def project_to_poincare(v, origin):
    """
    Projects a Euclidean vector v into a Poincaré Ball centered at the origin vector.
    """
    # 1. Translation: Shift the coordinate system so 'origin' is at [0, 0, ..., 0]
    v_shifted = v - origin
    
    # 2. Scaling: Normalize to the unit ball (radius < 1)
    # We use a safety epsilon to ensure the point is strictly inside the ball
    norm = np.linalg.norm(v_shifted)
    scale_factor = 0.99  # Max radius in Poincare ball
    
    if norm > 0:
        # Use log-dampening to prevent high-dimensional saturation (tanh(log1p(norm)))
        # This keeps 'close' Euclidean neighbors in the center of the ball
        # while pushing 'far' neighbors towards the boundary.
        dampened_norm = np.log1p(norm)
        v_poincare = (v_shifted / norm) * (np.tanh(dampened_norm) * scale_factor)
    else:
        v_poincare = v_shifted
        
    return v_poincare

def calculate_hyperbolic_radius(v_poincare):
    """
    Calculates the hyperbolic radius from the center of the Poincare ball.
    Formula: r = 2 * arctanh(||v||)
    """
    dist_norm = np.linalg.norm(v_poincare)
    # Clip to avoid infinity at the boundary
    dist_norm = np.clip(dist_norm, 0, 0.999)
    return 2 * np.arctanh(dist_norm)

def process_hyperbolic(input_file: str, output_file: str):
    print(f"Loading embeddings from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entity_embeddings = data['embeddings']['entities']
    
    print("Processing hyperbolic projections for each biography...")
    for record in data['records']:
        title = record['title'].lower()
        if title not in entity_embeddings:
            print(f"Warning: Title '{title}' not found in entity embeddings. Skipping.")
            continue
            
        origin_vec = np.array(entity_embeddings[title])
        
        for triple in record['triples']:
            sub_vec = np.array(entity_embeddings[triple['subject'].lower()])
            obj_vec = np.array(entity_embeddings[triple['object'].lower()])
            
            # Project subject and object relative to the biography's root (title)
            sub_poincare = project_to_poincare(sub_vec, origin_vec)
            obj_poincare = project_to_poincare(obj_vec, origin_vec)
            
            # Calculate radiuses
            triple['subject_radius'] = float(calculate_hyperbolic_radius(sub_poincare))
            triple['object_radius'] = float(calculate_hyperbolic_radius(obj_poincare))
            
            # Identity Score: Higher radius means further from the primary identity
            # (e.g. Spouse = Medium Radius, Subject = Near Zero Radius)
            triple['marginality_score'] = max(triple['subject_radius'], triple['object_radius'])

    print(f"Saving hyperbolic data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("Hyperbolic processing complete!")

if __name__ == "__main__":
    process_hyperbolic('refined_embeddings.json', 'hyperbolic_refined.json')
