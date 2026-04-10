import json
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

def generate_vectors(input_json, output_dir, model_name='nomic-ai/nomic-embed-text-v1.5'):
    """
    Extracts all unique entities from a KG JSON and generates embeddings.
    """
    input_path = Path(input_json)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading triples from {input_path}...")
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        # data = json.load(f)
        # Handle JSONL format
        data = [json.loads(line) for line in f if line.strip()]

    # Extract all unique strings (subjects, objects, and potentially relations)
    all_strings = set()
    for record in tqdm(data, desc="Extracting strings"):
        triples = record.get("triples", [])
        for t in triples:
            if t.get("subject"): all_strings.add(t["subject"].strip().lower())
            if t.get("object"): all_strings.add(t["object"].strip().lower())
            # Relations are often handled separately but can be included if needed
            if t.get("relation"): all_strings.add(t["relation"].strip().lower())

    sorted_strings = sorted(list(all_strings))
    print(f"Found {len(sorted_strings):,} unique strings.")

    # Save strings early
    strings_out = output_path / "kg_strings_44k.json"
    with open(strings_out, 'w', encoding='utf-8') as f:
        json.dump(sorted_strings, f, indent=2)
    print(f"Strings saved to {strings_out}")

    # Generate embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)

    print("Generating embeddings (this may take a while)...")
    # Batch processing for efficiency
    embeddings = model.encode(sorted_strings, 
                             batch_size=128, 
                             show_progress_bar=True, 
                             convert_to_tensor=True)

    vectors_out = output_path / "kg_vectors_44k.pt"
    torch.save(embeddings.cpu(), vectors_out)
    print(f"Vectors saved to {vectors_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vector embeddings for KG entities.")
    parser.add_argument("--input", type=str, default="processed_triples_44k.json", help="Input KG JSON file")
    parser.add_argument("--outdir", type=str, default="/scratch/INLP_Project_Wiki", help="Output directory for vectors and strings")
    parser.add_argument("--model", type=str, default="nomic-ai/nomic-embed-text-v1.5", help="SentenceTransformer model name")
    
    args = parser.parse_args()
    generate_vectors(args.input, args.outdir, args.model)
