import networkx as nx
import community.community_louvain as louvain
from networkx.algorithms import community

# 1. Load the SNAP Dataset
# Replace 'roadNet-CA.txt' with your specific file path
# SNAP files usually have comments starting with #
def load_snap_data(filepath):
    print(f"Loading dataset: {filepath}...")
    G = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def analyze_communities(G):
    # --- LOUVAIN METHOD ---
    print("\nRunning Louvain Algorithm...")
    partition_louvain = louvain.best_partition(G)
    mod_louvain = louvain.modularity(partition_louvain, G)
    print(f"Louvain Modularity: {mod_louvain:.4f}")

    # --- GREEDY MODULARITY (Greese) ---
    print("Running Greedy Modularity Algorithm...")
    # This returns a list of sets of nodes
    communities_greedy = community.greedy_modularity_communities(G)
    
    # NetworkX requires a slightly different approach to calculate modularity for this output
    mod_greedy = community.modularity(G, communities_greedy)
    print(f"Greedy Modularity: {mod_greedy:.4f}")

# Execute
path = "roadNet-CA.txt" 
G = load_snap_data(path)
analyze_communities(G)
