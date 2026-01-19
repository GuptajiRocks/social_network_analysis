import networkx as nx
import pandas as pd
import numpy as np

nodes = ["A", "B", "C"]

edges = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]

adl = {node : set() for node in nodes}

for i, j in edges:
    adl[i].add(j)
    adl[j].add(i)

print(adl)
