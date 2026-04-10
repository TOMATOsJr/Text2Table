import json
import os
from build_embeddings import create_embeddings
from hyperbolic_projection import process_hyperbolic
from sdp_grounding import process_sdp
from identity_filter import finalize_kg

def run_audit(limit=100):
    print(f"--- STARTING BATCH AUDIT (Limit: {limit} records) ---")
    
    # 1. Create a 100-record slice of output.json
    with open('output.json', 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    decoder = json.JSONDecoder()
    records = []
    pos = 0
    while len(records) < limit and pos < len(content):
        content = content.lstrip()
        if not content: break
        try:
            record, pos = decoder.raw_decode(content)
            records.append(record)
            content = content[pos:]
            pos = 0 
        except: break

    # Save temp input
    with open('audit_input.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    
    # 2. Run Pipeline
    print("\n[Audit] Generating embeddings...")
    create_embeddings('audit_input.json', 'audit_embeddings.json')
    
    print("\n[Audit] Calculating Hyperbolic Radii...")
    process_hyperbolic('audit_embeddings.json', 'audit_hyperbolic.json')
    
    print("\n[Audit] Grounding in Syntactic Paths...")
    process_sdp('audit_embeddings.json', 'audit_grounded.json') # Use base embeddings for SDP
    
    # We need to merge the hyperbolic and grounded data for the final filter
    with open('audit_hyperbolic.json', 'r', encoding='utf-8') as fh:
        hyp_data = json.load(fh)
    with open('audit_grounded.json', 'r', encoding='utf-8') as fg:
        gnd_data = json.load(fg)
        
    # Merge SDP paths into Hyperbolic dict
    for i, record in enumerate(hyp_data['records']):
        for j, triple in enumerate(record['triples']):
            triple['sdp_path'] = gnd_data['records'][i]['triples'][j].get('sdp_path', '')
            triple['sdp_status'] = gnd_data['records'][i]['triples'][j].get('sdp_status', 'N/A')
            
    with open('audit_composite.json', 'w', encoding='utf-8') as f:
        json.dump(hyp_data, f, indent=2)
        
    print("\n[Audit] Applying Hybrid Identity Filter...")
    finalize_kg('audit_composite.json', 'audit_final.json')
    
    # 3. Generate Report
    with open('audit_final.json', 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    total_triples = 0
    merged_triples = 0
    pruned_triples = 0
    grounded_count = 0
    
    for record in final_data['records']:
        for triple in record['triples']:
            total_triples += 1
            if triple['subject'] == record['title']:
                merged_triples += 1
            if triple.get('sdp_status') == "GROUNDED":
                grounded_count += 1
    
    # Note: finalized_kg.json only contains "clean" triples (merged/kept)
    # We compare back to audit_input.json to see pruning
    with open('audit_input.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    raw_total = sum(len(r.get('filtered_triples', [])) for r in raw_data)
    
    print("\n" + "="*40)
    print("      BIO-KG BATCH AUDIT REPORT       ")
    print("="*40)
    print(f"Processed Records:    {len(final_data['records'])}")
    print(f"Raw Extractions:      {raw_total}")
    print(f"Final KG Edges:       {total_triples}")
    print(f"Pruning Efficiency:   {((raw_total - total_triples)/raw_total)*100:.1f}%")
    print(f"Identity Merge Rate:  {(merged_triples/total_triples)*100:.1f}%")
    print(f"Syntactic Grounding:  {(grounded_count/total_triples)*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    run_audit(100)
