import numpy as np
import pandas as pd

nodes = ['A', 'B', 'C', 'D', 'E']

edges = [('A', 'B'), ('A', 'C'), ('C', 'E'), ('D', 'E'), ('B', 'C'), ('B', 'D')]

print("Forming Adjacency List")

adj_list = {node : [] for node in nodes} 
for i, j in edges:
    adj_list[i].append(j)
    adj_list[j].append(i)

for i in adj_list:
    print(f"{i} : {adj_list[i]}")

print(" ")

print("Forming Adjacency Matrix")

admat = [[0] * len(nodes) for blah in nodes]

for u, v in edges:
    m, n = nodes.index(u), nodes.index(v)
    admat[m][n] = 1
    admat[n][m] = 1

for i in admat:
    print(i)

print(" ")

print("Degree of Each Node")
for i in adj_list:
    print(f"Degree of Node {i} is -> {len(adj_list[i])}")

print(" ")

print("Graph Density")
size = len(nodes)

tedges = (sum(len(nbrs) for nbrs in adj_list.values())) / 2
gd = (2 * tedges) / (size * (size - 1))

print(f"Graph Density is: {gd}")
print(" ")

print("Clustering Coefficient Calculate")

for inp in nodes:
    nbs = adj_list[inp]
    k = len(nbs)
    apps = 0
    for i in range(k):
        for j in range(i+1, k):
            if nbs[j] in adj_list[nbs[i]]:
                apps+=1

    tlt = k * (k-1) / 2
    print(f"For Node -> {inp} the Clustering Coeff is -> {apps/tlt}")


