import time
import networkx as nx
from itertools import combinations
from collections import defaultdict
import community as community_louvain  # python-louvain

# =========================
# 1. LOAD HYPERGRAPH
# =========================
def load_hypergraph(file_path):
    hyperedges = []
    with open(file_path, 'r') as f:
        for line in f:
            nodes = list(map(int, line.strip().split(',')))
            if len(nodes) > 1:
                hyperedges.append(nodes)
    return hyperedges


# =========================
# 2. PROJECT TO GRAPH (CLIQUE EXPANSION)
# =========================
def hypergraph_to_graph(hyperedges):
    G = nx.Graph()
    
    for hedge in hyperedges:
        for u, v in combinations(hedge, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    
    return G


# =========================
# 3. COMMUNITY DETECTION (LOUVAIN)
# =========================
def detect_communities(G):
    partition = community_louvain.best_partition(G, weight='weight')
    
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    
    return list(communities.values()), partition


# =========================
# 4. STANDARD MODULARITY
# =========================
def compute_modularity(G, partition):
    return community_louvain.modularity(partition, G, weight='weight')


# =========================
# 5. OVERLAPPING MODULARITY (Approx)
# =========================
def overlapping_modularity(G, communities):
    m = G.size(weight='weight')
    Q = 0
    
    for community in communities:
        subgraph = G.subgraph(community)
        lc = subgraph.size(weight='weight')
        dc = sum(dict(G.degree(community, weight='weight')).values())
        
        if m > 0:
            Q += (lc / m) - (dc / (2*m))**2
    
    return Q


# =========================
# 6. EXTENDED MODULARITY (Hypergraph)
# =========================
def extended_modularity(hyperedges, communities):
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
    
    m = len(hyperedges)
    Q = 0
    
    for hedge in hyperedges:
        comm_count = defaultdict(int)
        
        for node in hedge:
            if node in node_to_comm:
                comm_count[node_to_comm[node]] += 1
        
        for c in comm_count:
            k = comm_count[c]
            Q += (k / len(hedge))**2
    
    return Q / m


# =========================
# 7. FULL PIPELINE
# =========================
def run_pipeline(file_path):
    start_time = time.time()
    
    hyperedges = load_hypergraph(file_path)
    G = hypergraph_to_graph(hyperedges)
    
    communities, partition = detect_communities(G)
    
    modularity = compute_modularity(G, partition)
    overlap_mod = overlapping_modularity(G, communities)
    ext_mod = extended_modularity(hyperedges, communities)
    
    end_time = time.time()
    
    print("\n===== RESULTS =====")
    print(f"Dataset: {file_path}")
    print(f"Communities found: {len(communities)}")
    print(f"Modularity: {modularity:.4f}")
    print(f"Overlapping Modularity: {overlap_mod:.4f}")
    print(f"Extended Modularity: {ext_mod:.4f}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")


# =========================
# 8. RUN ALL DATASETS
# =========================
if __name__ == "__main__":
    datasets = [
        "apr6_lab10/hyperedges-contact-high-school.txt",
        "apr6_lab10/hyperedges-contact-primary-school.txt",
        "apr6_lab10/hyperedges-house-bills.txt",
        "apr6_lab10/hyperedges-house-committees.txt",
        "apr6_lab10/hyperedges-senate-bills.txt",
        "apr6_lab10/hyperedges-senate-committees.txt",
        "apr6_lab10/hyperedges-trivago-clicks.txt",
        "apr6_lab10/hyperedges-walmart-trips.txt",
        "apr6_lab10/linear_large_edges_1000_he.txt",
    ]
    
    for dataset in datasets:
        run_pipeline(dataset)