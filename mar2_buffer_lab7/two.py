import networkx as nx
import community.community_louvain as louvain
from networkx.algorithms import community
import random
import time

def load_snap_data(filepath):
    print(f"Loading dataset: {filepath}...")
    G = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int, comments='#')
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def preprocess_graph(G, sample_size=100_000):
    """Get LCC then optionally sample it"""
    print("\nExtracting Largest Connected Component...")
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()
    print(f"LCC -> Nodes: {G_lcc.number_of_nodes()}, Edges: {G_lcc.number_of_edges()}")

    if G_lcc.number_of_nodes() > sample_size:
        print(f"Sampling {sample_size} nodes via BFS...")
        start = random.choice(list(G_lcc.nodes()))
        bfs_nodes = list(nx.bfs_tree(G_lcc, start).nodes())[:sample_size]
        G_lcc = G_lcc.subgraph(bfs_nodes).copy()
        print(f"Sampled -> Nodes: {G_lcc.number_of_nodes()}, Edges: {G_lcc.number_of_edges()}")

    return G_lcc

def analyze_communities(G):
    # --- LOUVAIN ---
    print("\nRunning Louvain Algorithm...")
    t = time.time()
    partition = louvain.best_partition(G, random_state=42)
    mod_louvain = louvain.modularity(partition, G)
    print(f"Louvain Modularity: {mod_louvain:.4f} | Time: {time.time()-t:.1f}s")
    print(f"Communities found: {len(set(partition.values()))}")

    # --- LABEL PROPAGATION (replaces slow Greedy) ---
    print("\nRunning Label Propagation Algorithm...")
    t = time.time()
    communities_lp = list(community.label_propagation_communities(G))
    mod_lp = community.modularity(G, communities_lp)
    print(f"Label Propagation Modularity: {mod_lp:.4f} | Time: {time.time()-t:.1f}s")
    print(f"Communities found: {len(communities_lp)}")

# Execute
path = "mar2_buffer_lab7/roadNet-CA.txt"
G = load_snap_data(path)
G = preprocess_graph(G, sample_size=100_000)  # Adjust sample size as needed
analyze_communities(G)
