"""
Compute node-level DINO cluster assignment for per-cluster topology.

Each structure point is assigned to a cluster via K-means on node_sem. Saved mask
is used by RealData as init_masks so the simulator builds topology with
per-cluster KNN + learnable edge gates (sigmoid).

Workflow (per-cluster topology + edge gating):
  1. Run extract_dino_semantic_features to get node_sem per case.
  2. Run this script to compute and save node_cluster per case:
       python semantic/compute_node_cluster.py --n_clusters 8
  3. Enable in config (e.g. configs/real.yaml or experiments_optimization):
       use_knn_topology: true
       use_edge_gating: true
       edge_gate_init: 3.0   # dense init: sigmoid(3) ~ 0.95
       object_knn: 30        # dense per-cluster candidate edges
  4. Run first-order optimization (trainer_warp): topology = per-cluster KNN,
     each edge has gate logit; optimizer learns gate (and log_a, spring_Y) so
     effective stiffness = sigmoid(gate) * exp(log_a) * k_base.

Usage (from project root):
    python semantic/compute_node_cluster.py --n_clusters 8
    python semantic/compute_node_cluster.py --case_name single_lift_cloth_4 --n_clusters 8

Output:
    semantic/cache/{case_name}_node_cluster.npy  [N] int, values in [0, n_clusters-1]
"""

import argparse
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from material_param_dataset import _load_structure_points


def main():
    parser = argparse.ArgumentParser(description="Compute DINO node clusters for per-cluster topology")
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--sem_cache_dir", type=str, default="semantic/cache")
    parser.add_argument("--experiments_optimization_dir", type=str, default="experiments_optimization")
    parser.add_argument("--case_to_material", type=str, default="semantic/case_to_material_different_types.json")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of DINO clusters (parts)")
    parser.add_argument("--case_name", type=str, default=None, help="Single case; if not set, process all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.sem_cache_dir, exist_ok=True)

    if args.case_name:
        case_names = [args.case_name]
    else:
        import json
        with open(args.case_to_material, "r") as f:
            raw = json.load(f)
        case_names = sorted(raw.get("case_to_material", raw).keys())

    for case_name in case_names:
        points_path = os.path.join(args.base_path, case_name, "final_data.pkl")
        sem_path = os.path.join(args.sem_cache_dir, f"{case_name}_node_sem.npz")
        if not os.path.isfile(points_path):
            print(f"[skip] {case_name}: no final_data.pkl")
            continue
        if not os.path.isfile(sem_path):
            print(f"[skip] {case_name}: no node_sem (run extract_dino_semantic_features first)")
            continue

        points = _load_structure_points(points_path)
        node_sem = np.load(sem_path)["node_sem"].astype(np.float32)
        if node_sem.shape[0] != points.shape[0]:
            print(f"[skip] {case_name}: node_sem shape {node_sem.shape[0]} != points {points.shape[0]}")
            continue

        N = node_sem.shape[0]
        n_clusters = min(args.n_clusters, N)
        if n_clusters < 2:
            print(f"[skip] {case_name}: too few points for clustering")
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
        labels = kmeans.fit_predict(node_sem)  # [N] int in [0, n_clusters-1]

        out_path = os.path.join(args.sem_cache_dir, f"{case_name}_node_cluster.npy")
        np.save(out_path, labels)
        print(f"[saved] {case_name}: {out_path} (N={N}, K={n_clusters})")


if __name__ == "__main__":
    main()
