import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import re
import argparse

# Default Paths
DEFAULT_VECTORS = Path("/scratch/INLP_Project_Wiki/kg_vectors_44k.pt")
DEFAULT_STRINGS = Path("/scratch/INLP_Project_Wiki/kg_strings_44k.json")
DEFAULT_INPUT = Path("/home2/pallamreddy.n/Project/processed_triples_44k_refined.json")
DEFAULT_OUTPUT = Path("/home2/pallamreddy.n/Project/processed_triples_44k_resolved.json")
DEFAULT_SYNONYMS = Path("/scratch/INLP_Project_Wiki/entity_merges.json")

def is_date(s):
    # Simple regex for Year or Month-Day-Year
    return bool(re.search(r'\d{4}', s))

def resolve_identities(input_path, vectors_path, strings_path, output_path, synonym_map_path):
    print(f"Loading vector cache from {vectors_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.load(vectors_path, map_location=device)
    
    with open(strings_path, 'r', encoding='utf-8') as f:
        all_strings = json.load(f)
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    num_nodes = len(all_strings)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    threshold = 0.96 
    batch_size = 5000
    
    print(f"Building similarity graph (Threshold: {threshold})...")
    for i in tqdm(range(0, num_nodes, batch_size), desc="Pairwise Matching"):
        end_i = min(i + batch_size, num_nodes)
        batch = embeddings[i:end_i]
        sim_matrix = torch.mm(batch, embeddings.t())
        matches = (sim_matrix >= threshold).nonzero()
        
        for match in matches:
            local_idx, global_idx = match.tolist()
            u, v = i + local_idx, global_idx
            if u == v: continue
            
            # --- CRITICAL PROTECTION: DATE COLLISION ---
            str_u, str_v = all_strings[u], all_strings[v]
            if is_date(str_u) or is_date(str_v):
                if str_u.lower() != str_v.lower():
                    continue
            G.add_edge(u, v)

    print("Finding connected components (Clusters)...")
    clusters = list(nx.connected_components(G))
    
    synonym_map = {}
    cluster_stats = []
    
    for cluster in tqdm(clusters, desc="Selecting Lead Nodes"):
        cluster_texts = [all_strings[idx] for idx in cluster]
        if len(cluster) == 1:
            synonym_map[cluster_texts[0]] = cluster_texts[0]
            continue
            
        lead_node = sorted(cluster_texts, key=lambda x: (len(x), sum(1 for c in x if c.isupper())), reverse=True)[0]
        
        for text in cluster_texts:
            synonym_map[text] = lead_node
        
        cluster_stats.append({"lead": lead_node, "synonyms": cluster_texts})

    print(f"Saving synonym map to {synonym_map_path}...")
    Path(synonym_map_path).parent.mkdir(parents=True, exist_ok=True)
    with open(synonym_map_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, indent=2)

    # Phase 2: Update the Knowledge Graph
    print(f"Updating {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    resolved_records = []
    for record in tqdm(records, desc="Merging Records"):
        triples = record.get("triples", [])
        new_triples = []
        seen_triples = set()
        for t in triples:
            subj = synonym_map.get(t["subject"].strip().lower(), t["subject"])
            obj = synonym_map.get(t["object"].strip().lower(), t["object"])
            rel = t["relation"]
            t_key = (subj.lower(), rel.lower(), obj.lower())
            if t_key not in seen_triples:
                new_triples.append({"subject": subj, "relation": rel, "object": obj})
                seen_triples.add(t_key)
        record["triples"] = new_triples
        resolved_records.append(record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resolved_records, f, indent=2)
    print(f"Success: Resolved KG saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Identity Resolution for Knowledge Graphs.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--vectors", type=str, default=str(DEFAULT_VECTORS))
    parser.add_argument("--strings", type=str, default=str(DEFAULT_STRINGS))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--synonyms", type=str, default=str(DEFAULT_SYNONYMS))
    
    args = parser.parse_args()
    resolve_identities(args.input, args.vectors, args.strings, args.output, args.synonyms)
