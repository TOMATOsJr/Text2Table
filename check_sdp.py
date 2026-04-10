import json

def check_sdp(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i in range(2):
        record = data['records'][i]
        print(f"\n=== {record['title'].upper()} (Grounding Audit) ===")
        
        for t in record['triples'][:5]:
            status = t.get('sdp_status', 'N/A')
            path = t.get('sdp_path', 'N/A')
            print(f"Status: {status:12} | Path: {path}")
            print(f"   Triple: ({t['subject']}) --[{t['relation']}]--> ({t['object']})")

if __name__ == "__main__":
    check_sdp('grounded_triples.json')
