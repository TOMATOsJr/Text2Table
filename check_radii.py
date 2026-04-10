import json

def check_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i in range(2):  # Check first 2 biographies
        record = data['records'][i]
        print(f"\n=== {record['title'].upper()} (Origin) ===")
        
        # Sort triples by marginality to see who is closest vs furthest
        sorted_triples = sorted(record['triples'], key=lambda x: x.get('marginality_score', 0))
        
        for t in sorted_triples[:8]:  # Top 8 most 'central' triples
            print(f"R={t['marginality_score']:.4f} | {t['subject']} -> {t['relation']} -> {t['object']}")

if __name__ == "__main__":
    check_results('hyperbolic_refined.json')
