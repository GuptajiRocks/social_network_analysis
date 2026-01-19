import networkx as nx

# Create a simple Graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])

nodes_to_check = [1, 2, 3]

# Verify the clique
is_clique = nx.find_cliques(G)
print(is_clique)
