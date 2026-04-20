import networkx as nx
import numpy as np
import time
from collections import defaultdict


# ─────────────────────────────────────────────
#  MODULARITY  (optimized)
# ─────────────────────────────────────────────

def modu1_fast(G, U, N, O, node_to_idx):
    """
    O(n²) modularity for small communities (<= 100 nodes).
    - O(1) node→index lookup via precomputed dict
    - degrees and overlaps cached upfront
    - edge set for O(1) has_edge
    """
    deg     = dict(G.degree(U))
    overlap = {u: O[node_to_idx[u]] for u in U}
    edge_set = set(G.edges())

    def has_edge(a, b):
        return (a, b) in edge_set or (b, a) in edge_set

    n     = len(U)
    m     = 0.0
    two_N = 2.0 * N

    for i in range(n):
        ui = U[i]
        b1 = overlap[ui]
        di = deg[ui]
        for j in range(i, n):
            uj     = U[j]
            b2     = overlap[uj]
            dj     = deg[uj]
            w      = 1.0 / (b1 * b2)
            dterm  = (di * dj) / two_N

            if i == j:
                m += w * (-dterm)
            elif has_edge(ui, uj):
                m += 2.0 * w * (1.0 - dterm)
            else:
                m += 2.0 * w * (-dterm)

    return m


def modu1_numpy(G, U, N, O, node_to_idx):
    """
    Vectorised numpy modularity for large communities (> 100 nodes).
    """
    deg     = np.array([G.degree(u) for u in U], dtype=np.float64)
    overlap = np.array([O[node_to_idx[u]] for u in U], dtype=np.float64)

    sub = G.subgraph(U)
    adj = nx.to_numpy_array(sub, nodelist=U)

    w        = np.outer(1.0 / overlap, 1.0 / overlap)
    deg_prod = np.outer(deg, deg) / (2.0 * N)

    base       = -w * deg_prod
    edge_bonus = w * adj
    M          = 2.0 * (base + edge_bonus)

    # diagonal counted once, not twice
    np.fill_diagonal(M, -w.diagonal() * np.diag(deg_prod))

    return float(M.sum())


def compute_modularity(G, U, N, O, node_to_idx):
    """Dispatcher: numpy for large communities, pure-Python for small."""
    if len(U) > 100:
        return modu1_numpy(G, U, N, O, node_to_idx)
    return modu1_fast(G, U, N, O, node_to_idx)


# ─────────────────────────────────────────────
#  COMMUNITY DETECTION  (optimized)
# ─────────────────────────────────────────────

def b(path, sep):
    # ── Load graph ──────────────────────────────
    G = nx.read_edgelist(path, comments='#', delimiter=sep,
                         nodetype=int, encoding='utf-8')
    print(f"Loaded  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

    ns  = G.number_of_nodes()
    N   = G.number_of_edges()
    den = nx.density(G)
    etr = den * ns
    print(f"density={den:.6f}")

    # threshold tuning (kept from original logic)
    th = 0.7 if etr > 7 else 0.6
    th = 0.5   # original hard-override kept

    # ── Precompute neighbour sets (big speedup for intersection ops) ──
    nbr = {u: set(G.neighbors(u)) for u in G.nodes()}

    tsp1 = time.time()

    res = []
    tr  = list(G.nodes())   # unprocessed nodes

    # ── Phase 1 : seed + expand ──────────────────
    while tr:
        i1 = tr[0]
        xx = list(nbr[i1])

        # find neighbour sharing most common neighbours with i1
        mx, tp = 0, -1
        for j, nb in enumerate(xx):
            o = len(nbr[i1] & nbr[nb])
            if o > mx:
                mx, tp = o, j

        if mx == 0:
            tr.pop(0)
            continue

        # seed community
        c   = [i1, xx[tp]]
        c_set = set(c)

        # candidate pool = neighbours of seed nodes, not yet in c
        T = list((nbr[c[0]] | nbr[c[1]]) - c_set)

        # expand in threshold passes
        for threshold in (0.9, 0.8, th):
            still_in = []
            for k in T:
                cpt = len(nbr[k] & c_set)
                if cpt / len(c_set) >= threshold:
                    c.append(k)
                    c_set.add(k)
                else:
                    still_in.append(k)
            T = still_in

        # final pass: node whose majority of its own edges go inside c
        for k in T:
            rr = nbr[k]
            if len(rr & c_set) / len(rr) > 0.5:
                c.append(k)
                c_set.add(k)

        # remove absorbed nodes from the pool
        tr_set = set(tr) - c_set
        nn2    = len(tr_set)
        n2     = len(tr) - nn2
        por    = n2 / len(c)
        tr     = list(tr_set)

        if por >= 0.5:
            res.append(c)
        else:
            # try to merge into an existing community
            merged = False
            for i3, existing in enumerate(res):
                existing_set = set(existing)
                if len(existing_set & c_set) >= min(len(existing), len(c)) / 2:
                    res[i3] = list(existing_set | c_set)
                    merged = True
                    break
            if not merged:
                res.append(c)

    # ── Phase 2 : assign leftover nodes ──────────
    sup = tr   # any nodes still unprocessed
    for k in sup:
        ne  = nbr[k]
        best_i, best_cnt = -1, 0
        for i, community in enumerate(res):
            aa = len(ne & set(community))
            if aa >= best_cnt:
                best_cnt = aa
                best_i   = i
                if aa > len(ne) - aa:
                    break
        if best_cnt > 0:
            res[best_i].append(k)

    print(f"Communities before merge: {len(res)}")

    # ── Phase 3 : merge overlapping communities ──
    res.sort(key=len, reverse=True)
    r = 0
    while r < len(res):
        j = r + 1
        while j < len(res):
            if len(set(res[r]) & set(res[j])) >= len(res[j]) / 3:
                res[r] = list(set(res[r]) | set(res[j]))
                res.pop(j)
            else:
                j += 1
        r += 1

    tsp2 = time.time()
    print(f"Communities after merge : {len(res)}")
    print(f"Detection time          : {tsp2 - tsp1:.2f}s")

    # ── Write results ────────────────────────────
    with open("res.txt", "w") as fichier:
        for comm in res:
            fichier.write('\t'.join(str(k) for k in comm) + '\n')

    # ── Build overlap count vector O ─────────────
    node_list   = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    O  = [0] * ns
    re = []
    for comm in res:
        re.extend(comm)
    for node in re:
        O[node_to_idx[node]] += 1

    # Nodes appearing in >1 community
    overlapping = [node for node in G.nodes() if O[node_to_idx[node]] > 1]
    print(f"Overlapping nodes       : {len(overlapping)}")

    # ── Compute modularity ───────────────────────
    print("Computing modularity...")
    t_mod = time.time()
    m = 0.0
    for comm in res:
        m += compute_modularity(G, list(comm), N, O, node_to_idx)
    m /= (2.0 * N)
    print(f"Modularity              : {m:.6f}")
    print(f"Modularity time         : {time.time() - t_mod:.2f}s")

    return m, res


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    path = "mar2_buffer_lab7/roadNet-CA.txt"
    sep  = "\t"          # change to "\t" or "," if needed
    modularity, communities = b(path, sep)
