import random
from collections import deque, defaultdict

import networkx as nx
import community as community_louvain  # python-louvain
from networkx.algorithms.community.quality import modularity


def load_edgelist_txt(path, delimiter=None, nodetype=int, directed=False):
    """
    Robust edgelist loader:
    - handles space/tab/comma separated files
    - skips headers like "source,target"
    - supports nodetype=int or nodetype=str
    """
    import gzip, re
    opener = gzip.open if path.endswith(".gz") else open
    G = nx.DiGraph() if directed else nx.Graph()

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            # delimiter auto-detect if delimiter not provided
            if delimiter is None:
                # split on comma OR whitespace OR tabs
                parts = re.split(r"[,\t\s]+", line)
            else:
                parts = line.split(delimiter)

            if len(parts) < 2:
                continue

            # handle possible header row
            try:
                u = nodetype(parts[0])
                v = nodetype(parts[1])
            except Exception:
                continue

            G.add_edge(u, v)

    return G


def load_gml(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # remove problematic _pos lines
    cleaned = [line for line in lines if "_pos" not in line]

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.writelines(cleaned)
        tmp_path = tmp.name

    return nx.read_gml(tmp_path, label="id")


def load_csv_edgelist(path, src_col=0, dst_col=1, weight_col=None, sep=","):
    import pandas as pd
    df = pd.read_csv(path, sep=sep)
    G = nx.Graph()
    if weight_col is None:
        for _, r in df.iterrows():
            G.add_edge(int(r.iloc[src_col]), int(r.iloc[dst_col]))
    else:
        for _, r in df.iterrows():
            G.add_edge(int(r.iloc[src_col]), int(r.iloc[dst_col]), weight=float(r.iloc[weight_col]))
    return G


def largest_cc(G):
    if G.is_directed():
        H = G.to_undirected()
    else:
        H = G
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    if not comps:
        return G
    nodes = comps[0]
    return G.subgraph(nodes).copy()


def jaccard_on_edge(G, u, v):
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    inter = len(Nu & Nv)
    uni = len(Nu | Nv)
    return 0.0 if uni == 0 else inter / uni


def build_ic_probabilities(G, beta=0.7):
    """
    p(u,v) = min(1, beta * jaccard(u,v) + (1-beta)*(1/deg(u)))
    """
    p = {u: {} for u in G.nodes()}
    for u, v in G.edges():
        s = jaccard_on_edge(G, u, v)
        puv = min(1.0, beta * s + (1 - beta) * (1.0 / max(1, G.degree(u))))
        pvu = min(1.0, beta * s + (1 - beta) * (1.0 / max(1, G.degree(v))))
        p[u][v] = puv
        p[v][u] = pvu
    return p


def select_seeds_top_degree(G, k=30):
    deg_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [u for u, _ in deg_sorted[:k]]


def ic_multisource_labels(G, p, seeds):
    """
    Multi-source IC:
    - start with all seeds active
    - newly active node gets ONE chance to activate neighbors
    - label propagates from seed
    """
    label = {s: s for s in seeds}
    active = set(seeds)
    q = deque(seeds)

    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v in active:
                continue
            if random.random() <= p[u].get(v, 0.0):
                active.add(v)
                label[v] = label[u]
                q.append(v)
    return label


def diffusion_edge_weights(G, p, seeds, R=50, seed=42):
    random.seed(seed)
    same = defaultdict(int)

    for _ in range(R):
        lab = ic_multisource_labels(G, p, seeds)
        for u, v in G.edges():
            if (u in lab) and (v in lab) and (lab[u] == lab[v]):
                same[(u, v)] += 1

    wprime = {}
    for u, v in G.edges():
        wprime[(u, v)] = same[(u, v)] / R
    return wprime


def run_ic_mod(G, beta=0.7, lam=0.8, k=30, R=50, seed=42, use_lcc=True):
    if use_lcc:
        G = largest_cc(G)

    random.seed(seed)

    seeds = select_seeds_top_degree(G, k=k)
    p = build_ic_probabilities(G, beta=beta)
    wprime = diffusion_edge_weights(G, p, seeds, R=R, seed=seed)

    Gw = nx.Graph()
    Gw.add_nodes_from(G.nodes())

    for u, v in G.edges():
        base_w = 1.0
        w = lam * wprime[(u, v)] + (1 - lam) * base_w
        Gw.add_edge(u, v, weight=w)

    part = community_louvain.best_partition(Gw, weight="weight", random_state=seed)

    comm_map = {}
    for node, cid in part.items():
        comm_map.setdefault(cid, set()).add(node)
    communities = list(comm_map.values())

    Q = modularity(Gw, communities, weight="weight")
    return Q, communities, seeds, Gw