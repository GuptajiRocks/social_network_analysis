import argparse
import os
import csv
import time
import networkx as nx

from icmod import (
    load_edgelist_txt,
    load_gml,
    load_csv_edgelist,
    run_ic_mod
)


def save_partition(out_path, communities):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(communities):
            for node in c:
                f.write(f"{node},{i}\n")


def main():
    parser = argparse.ArgumentParser(description="IC-MOD: Independent Cascade based community detection")
    parser.add_argument("--name", required=True, help="Dataset name for reporting")
    parser.add_argument("--input", required=True, help="Path to dataset file")
    parser.add_argument("--format", required=True, choices=["snap", "gml", "csv"],
                        help="snap: edgelist txt(.gz), gml: .gml, csv: .csv")
    parser.add_argument("--directed", action="store_true", help="Treat input as directed (snap only)")
    parser.add_argument("--beta", type=float, default=0.7, help="IC prob mix weight")
    parser.add_argument("--lam", type=float, default=0.8, help="Diffusion weight mix")
    parser.add_argument("--k", type=int, default=30, help="Number of seeds")
    parser.add_argument("--R", type=int, default=50, help="Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_lcc", action="store_true", help="Disable largest connected component extraction")
    parser.add_argument("--results", default="results", help="Results folder")
    parser.add_argument("--nodetype", default="int", choices=["int", "str"], help="Node id type conversion: int (default) or str")

    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(os.path.join(args.results, "partitions"), exist_ok=True)

    # Load graph
    if args.format == "snap":
        G = load_edgelist_txt(args.input, delimiter=None, nodetype=int, directed=args.directed)
    elif args.format == "gml":
        G = load_gml(args.input)
        # ensure simple undirected graph
        if G.is_directed():
            G = G.to_undirected()
    else:  # csv
        # assumes first two columns are src,dst
        G = load_csv_edgelist(args.input, src_col=0, dst_col=1, weight_col=None, sep=",")

    t0 = time.time()
    Q, communities, seeds, Gw = run_ic_mod(
        G,
        beta=args.beta,
        lam=args.lam,
        k=args.k,
        R=args.R,
        seed=args.seed,
        use_lcc=(not args.no_lcc)
    )
    t1 = time.time()

    # Save partition
    part_path = os.path.join(args.results, "partitions", f"{args.name}_partition.csv")
    save_partition(part_path, communities)

    # Append metrics
    metrics_path = os.path.join(args.results, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset", "nodes", "edges", "communities", "modularity_Q", "time_sec", "k", "R", "beta", "lam"])
        w.writerow([
            args.name,
            Gw.number_of_nodes(),
            Gw.number_of_edges(),
            len(communities),
            round(Q, 6),
            round(t1 - t0, 3),
            args.k,
            args.R,
            args.beta,
            args.lam
        ])

    print("DONE ✅")
    print(f"Dataset: {args.name}")
    print(f"Nodes: {Gw.number_of_nodes()}  Edges: {Gw.number_of_edges()}")
    print(f"Communities: {len(communities)}")
    print(f"Modularity Q: {Q:.6f}")
    print(f"Time (sec): {t1 - t0:.3f}")
    print(f"Partition saved: {part_path}")
    print(f"Metrics appended: {metrics_path}")


if __name__ == "__main__":
    main()