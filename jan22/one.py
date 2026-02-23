import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_edges_from([
    ("A","B"),
    ("A","C"),
    ("B","C"),
    ("B","D"),
    ("C","D"),
    ("C","E"),
    ("D","E"),
    ("E","F"),
    ("D","F")
])

for u, v in G.edges():
    G[u][v]["weight"] = 1.0

adamic_scores = nx.adamic_adar_index(G)

for u, v, score in adamic_scores:
    if score > 0:
        G.add_edge(u, v, weight=score)

pos = nx.spring_layout(G, seed=42)

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

nx.draw(G, pos, with_labels=True, node_size=2000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
