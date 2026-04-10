import json
import networkx as nx
import matplotlib.pyplot as plt
from knowledge_graph import build_knowledge_graph, print_knowledge_graph

def create_hybrid_kg(input_json_path, visualize=True):
    """
    Reads hybrid extraction results and builds a Knowledge Graph.
    """
    print(f"Loading extractions from {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_json_path} not found. Please ensure you have the hybrid output file.")
        return

    # Convert dictionary format to tuples for the KG builder
    all_triples = []
    for entry in data:
        for t in entry.get('triples', []):
            all_triples.append((t['subject'], t['relation'], t['object']))
    
    # 1. Build the logical KG
    kg_dict = build_knowledge_graph(all_triples)
    
    # 2. Print the text-based KG
    print("\n--- Knowledge Graph Structure ---")
    print_knowledge_graph(kg_dict)
    
    # 3. Optional: NetworkX Visualization
    if visualize:
        print("\nGenerating visualization...")
        G = nx.MultiDiGraph()
        
        # Add a limit for visualization to prevent clutter
        for subj, rel, obj in all_triples[:50]:
            G.add_edge(subj, obj, label=rel)
            
        plt.figure(figsize=(15, 10))
        # Use shell_layout or kamada_kawai for more spread out results
        pos = nx.kamada_kawai_layout(G)
        
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightgreen', 
                node_size=3000, 
                edge_color='silver', 
                font_size=12, 
                font_weight='bold',
                arrows=True,
                arrowsize=20)
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=10)
        
        plt.title("Hybrid Knowledge Graph (Top 50 Triplets)", fontsize=16)
        
        # Save the visualization to a file
        output_image = "knowledge_graph.png"
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_image}")
        plt.show()

if __name__ == "__main__":
    # Path to your hybrid output file
    INPUT_FILE = "hybrid_extractions.json"
    create_hybrid_kg(INPUT_FILE)
