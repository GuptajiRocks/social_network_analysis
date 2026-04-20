import networkx as nx
import numpy as np
import time


# ─────────────────────────────────────────────
#  BITSET UTILITIES
# ─────────────────────────────────────────────

def build_bitsets(G):
    """
    Assign each node a unique bit position.
    Return:
      node_to_bit : dict  node -> int  (e.g. node 5 -> 1<<3)
      nbr_bits    : dict  node -> int  (bitmask of all neighbours)
      all_nodes   : list  (stable ordering)
    """
    all_nodes  = list(G.nodes())
    node_to_bit = {n: (1 << i) for i, n in enumerate(all_nodes)}

    nbr_bits = {}
    for n in all_nodes:
        mask = 0
        for nb in G.neighbors(n):
            mask |= node_to_bit[nb]
        nbr_bits[n] = mask

    return node_to_bit, nbr_bits, all_nodes


def bits_to_nodes(mask, all_nodes, node_to_bit):
    """Convert a bitmask back to a list of nodes."""
    return [n for n in all_nodes if mask & node_to_bit[n]]


def popcount(mask):
    """Count set bits (number of nodes in a bitmask community)."""
    return bin(mask).count('1')


# ─────────────────────────────────────────────
#  MODULARITY  (bitset-accelerated)
# ─────────────────────────────────────────────

def modu1_bitset(G, comm_mask, N, O, node_to_bit, node_to_idx, all_nodes):
    """
    Modularity using bitsets for adjacency.
    - comm_mask : bitmask of nodes in this community
    - Intersection via & gives shared neighbours instantly
    """
    U      = bits_to_nodes(comm_mask, all_nodes, node_to_bit)
    n      = len(U)
    two_N  = 2.0 * N
    m      = 0.0

    # Precompute per-node values
    deg     = {u: G.degree(u)            for u in U}
    overlap = {u: O[node_to_idx[u]]      for u in U}
    # Bitmask of each node's neighbours, restricted to community
    nbr_in_comm = {u: G_nbr_bits[u] & comm_mask for u in U}

    for i in range(n):
        ui = U[i]
        b1 = overlap[ui]
        di = deg[ui]
        ui_bit = node_to_bit[ui]

        for j in range(i, n):
            uj    = U[j]
            b2    = overlap[uj]
            dj    = deg[uj]
            w     = 1.0 / (b1 * b2)
            dterm = (di * dj) / two_N

            if i == j:
                m += w * (-dterm)
            # Check adjacency via bitset: is ui in uj's neighbour mask?
            elif nbr_in_comm[uj] & ui_bit:
                m += 2.0 * w * (1.0 - dterm)
            else:
                m += 2.0 * w * (-dterm)

    return m


def modu1_numpy(G, U, N, O, node_to_idx):
    """Vectorised numpy for large communities."""
    deg     = np.array([G.degree(u) for u in U], dtype=np.float64)
    overlap = np.array([O[node_to_idx[u]] for u in U], dtype=np.float64)
    sub     = G.subgraph(U)
    adj     = nx.to_numpy_array(sub, nodelist=U)
    w       = np.outer(1.0 / overlap, 1.0 / overlap)
    deg_prod = np.outer(deg, deg) / (2.0 * N)
    base     = -w * deg_prod
    M        = 2.0 * (base + w * adj)
    np.fill_diagonal(M, -w.diagonal() * np.diag(deg_prod))
    return float(M.sum())


def compute_modularity(G, comm_mask, N, O, node_to_bit, node_to_idx, all_nodes):
    U = bits_to_nodes(comm_mask, all_nodes, node_to_bit)
    if len(U) > 100:
        return modu1_numpy(G, U, N, O, node_to_idx)
    return modu1_bitset(G, comm_mask, N, O, node_to_bit, node_to_idx, all_nodes)


# ─────────────────────────────────────────────
#  COMMUNITY DETECTION  (bitset-accelerated)
# ─────────────────────────────────────────────

# Global so modularity can access it without passing everywhere
G_nbr_bits = {}

def b(path, sep):
    global G_nbr_bits

    # ── Load ────────────────────────────────────
    G = nx.read_edgelist(path, comments='#', delimiter=sep,
                         nodetype=int, encoding='utf-8')
    print(f"Loaded  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

    ns  = G.number_of_nodes()
    N   = G.number_of_edges()
    den = nx.density(G)
    etr = den * ns
    print(f"density={den:.6f}")

    th = 0.7 if etr > 7 else 0.6
    th = 0.5  # original hard-override

    # ── Build bitsets ────────────────────────────
    print("Building bitsets...")
    node_to_bit, G_nbr_bits, all_nodes = build_bitsets(G)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    print("Bitsets ready.")

    tsp1 = time.time()

    # Work with bitmasks throughout — no list copies needed
    res_masks = []                            # list of community bitmasks
    remaining = sum(node_to_bit.values())     # bitmask of all unprocessed nodes

    # ── Phase 1 : seed + expand ──────────────────
    while remaining:
        # Pick lowest set bit = next unprocessed node
        i1 = all_nodes[remaining.bit_length() - 1
                        if False else
                        (remaining & -remaining).bit_length() - 1]

        # Actually: find first set bit properly
        lsb  = remaining & (-remaining)          # isolates lowest set bit
        i1   = all_nodes[lsb.bit_length() - 1]  # map bit position → node

        nbr_mask = G_nbr_bits[i1]
        xx = bits_to_nodes(nbr_mask, all_nodes, node_to_bit)

        if not xx:
            remaining &= ~lsb
            continue

        # Find neighbour with most common neighbours (triangle count)
        mx, best_nb = 0, None
        for nb in xx:
            common = popcount(G_nbr_bits[i1] & G_nbr_bits[nb])
            if common > mx:
                mx, best_nb = common, nb

        if mx == 0:
            remaining &= ~lsb
            continue

        # Seed community as bitmask
        c_mask  = node_to_bit[i1] | node_to_bit[best_nb]
        c_size  = 2

        # Candidate pool: union of seed neighbours, minus seed itself
        T_mask = (G_nbr_bits[i1] | G_nbr_bits[best_nb]) & ~c_mask

        # Expand: threshold passes
        for threshold in (0.9, 0.8, th):
            add_mask  = 0
            keep_mask = 0
            # iterate candidates
            t_remaining = T_mask
            while t_remaining:
                k_bit = t_remaining & (-t_remaining)
                k     = all_nodes[k_bit.bit_length() - 1]
                # intersection of k's neighbours with community = popcount of AND
                cpt   = popcount(G_nbr_bits[k] & c_mask)
                if cpt / c_size >= threshold:
                    add_mask |= k_bit
                else:
                    keep_mask |= k_bit
                t_remaining &= t_remaining - 1   # clear lowest bit

            c_mask  |= add_mask
            c_size   = popcount(c_mask)
            T_mask   = keep_mask

        # Final pass: majority of node's own edges go inside community
        t_remaining = T_mask
        while t_remaining:
            k_bit = t_remaining & (-t_remaining)
            k     = all_nodes[k_bit.bit_length() - 1]
            k_deg = G.degree(k)
            if k_deg > 0:
                inside = popcount(G_nbr_bits[k] & c_mask)
                if inside / k_deg > 0.5:
                    c_mask |= k_bit
                    c_size += 1
            t_remaining &= t_remaining - 1

        # Remove absorbed nodes from remaining pool
        absorbed  = remaining & c_mask
        n2        = popcount(absorbed)
        por       = n2 / c_size
        remaining &= ~c_mask

        if por >= 0.5:
            res_masks.append(c_mask)
        else:
            merged = False
            for i3 in range(len(res_masks)):
                overlap_bits = popcount(res_masks[i3] & c_mask)
                min_size     = min(popcount(res_masks[i3]), c_size)
                if overlap_bits >= min_size / 2:
                    res_masks[i3] |= c_mask
                    merged = True
                    break
            if not merged:
                res_masks.append(c_mask)

    # ── Phase 2 : assign leftover nodes ──────────
    # (remaining is now only truly isolated nodes)
    leftover = remaining
    while leftover:
        k_bit  = leftover & (-leftover)
        k      = all_nodes[k_bit.bit_length() - 1]
        k_nbrs = G_nbr_bits[k]
        best_i, best_cnt = -1, 0
        for i, cm in enumerate(res_masks):
            cnt = popcount(k_nbrs & cm)
            if cnt > best_cnt:
                best_cnt = cnt
                best_i   = i
        if best_i >= 0:
            res_masks[best_i] |= k_bit
        leftover &= leftover - 1

    print(f"Communities before merge: {len(res_masks)}")

    # ── Phase 3 : merge overlapping communities ──
    res_masks.sort(key=popcount, reverse=True)
    r = 0
    while r < len(res_masks):
        j = r + 1
        while j < len(res_masks):
            if popcount(res_masks[r] & res_masks[j]) >= popcount(res_masks[j]) / 3:
                res_masks[r] |= res_masks[j]
                res_masks.pop(j)
            else:
                j += 1
        r += 1

    tsp2 = time.time()
    print(f"Communities after merge : {len(res_masks)}")
    print(f"Detection time          : {tsp2 - tsp1:.2f}s")

    # ── Write results ────────────────────────────
    with open("res.txt", "w") as f:
        for cm in res_masks:
            nodes = bits_to_nodes(cm, all_nodes, node_to_bit)
            f.write('\t'.join(str(k) for k in nodes) + '\n')

    # ── Overlap count vector O ───────────────────
    O = [0] * ns
    for cm in res_masks:
        for n in bits_to_nodes(cm, all_nodes, node_to_bit):
            O[node_to_idx[n]] += 1

    overlapping = sum(1 for i in range(ns) if O[i] > 1)
    print(f"Overlapping nodes       : {overlapping}")

    # ── Modularity ───────────────────────────────
    print("Computing modularity...")
    t_mod = time.time()
    m = 0.0
    for cm in res_masks:
        m += compute_modularity(G, cm, N, O, node_to_bit, node_to_idx, all_nodes)
    m /= (2.0 * N)
    print(f"Modularity              : {m:.6f}")
    print(f"Modularity time         : {time.time() - t_mod:.2f}s")

    return m, res_masks


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    path = "roadNet-CA.txt"
    sep  = "\t"
    modularity, communities = b(path, sep)
