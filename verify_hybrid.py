import json

def verify_hybrid(file_path):
    print(f"Opening {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            print(f"Total entries: {len(data)}")
            
            # Count entries with triples
            count_with_triples = sum(1 for entry in data if entry.get('triples'))
            print(f"Entries with triples: {count_with_triples}")
            
            if count_with_triples > 0:
                # Show first entry with triples
                for entry in data:
                    if entry.get('triples'):
                        print("\nFirst Entry with Triples Sample:")
                        print(f"Text Snippet: {entry.get('text', '')[:100]}...")
                        print(f"Triples: {entry['triples'][:2]}")
                        break
            else:
                print("No triples found in this file.")
                
        except Exception as e:
            print(f"Error loading JSON: {e}")

if __name__ == "__main__":
    verify_hybrid('hybrid_extractions.json')
