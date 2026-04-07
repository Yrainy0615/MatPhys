import argparse
import json
import os
import pickle
from collections import defaultdict
import sys
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaussian_splatting.utils.read_write_model import read_cameras_binary, read_images_binary


BRANCH_LABEL = 0
LEAF_LABEL = 1

EDGE_BRANCH_CHAIN = 0
EDGE_BRANCH_SUPPORT = 1
EDGE_LEAF_INTRA = 2
EDGE_LEAF_ATTACH = 3


def _farthest_point_sample_indices(
    points: np.ndarray, n_samples: int, seed: int = 0
) -> np.ndarray:
    n_points = int(points.shape[0])
    if n_samples >= n_points:
        return np.arange(n_points, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = np.zeros(n_samples, dtype=np.int64)
    selected[0] = int(rng.integers(0, n_points))

    dist2 = np.full(n_points, np.inf, dtype=np.float32)
    for i in range(1, n_samples):
        last_pt = points[selected[i - 1]]
        cur_dist2 = np.sum((points - last_pt[None, :]) ** 2, axis=1)
        dist2 = np.minimum(dist2, cur_dist2)
        selected[i] = int(np.argmax(dist2))
    return np.unique(selected)


def _build_sampled_knn_topology(
    points: np.ndarray,
    labels: np.ndarray,
    knn: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = int(points.shape[0])
    if n_points < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    tree = cKDTree(points)
    k_eff = min(max(int(knn), 1), n_points - 1)
    dists, nn_idx = tree.query(points, k=k_eff + 1)

    rows = []
    cols = []
    vals = []
    edges = []
    edge_types = []
    for i, (row_d, row_j) in enumerate(zip(dists, nn_idx)):
        for dist, j in zip(np.atleast_1d(row_d)[1:], np.atleast_1d(row_j)[1:]):
            j = int(j)
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([float(dist), float(dist)])
            edges.append([i, j])
            if labels[i] == BRANCH_LABEL and labels[j] == BRANCH_LABEL:
                edge_types.append(EDGE_BRANCH_CHAIN)
            elif labels[i] == LEAF_LABEL and labels[j] == LEAF_LABEL:
                edge_types.append(EDGE_LEAF_INTRA)
            else:
                edge_types.append(EDGE_LEAF_ATTACH)

    # Add an MST to guarantee global connectivity across the sampled nodes.
    graph = csr_matrix((vals, (rows, cols)), shape=(n_points, n_points))
    mst = minimum_spanning_tree(graph).tocoo()
    for i, j in zip(mst.row.tolist(), mst.col.tolist()):
        edges.append([int(i), int(j)])
        if labels[i] == BRANCH_LABEL and labels[j] == BRANCH_LABEL:
            edge_types.append(EDGE_BRANCH_SUPPORT)
        elif labels[i] == LEAF_LABEL and labels[j] == LEAF_LABEL:
            edge_types.append(EDGE_LEAF_INTRA)
        else:
            edge_types.append(EDGE_LEAF_ATTACH)

    return _dedupe_edges(edges, edge_types)


def _read_ply_xyz_rgb(path: str):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb = None
    if {"red", "green", "blue"}.issubset(vertex.data.dtype.names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(
            np.float32
        )
        if rgb.max() > 1.0:
            rgb /= 255.0
    return xyz, rgb, ply


def _read_ply_edges(ply: PlyData) -> np.ndarray:
    if "edge" not in {element.name for element in ply.elements}:
        return np.zeros((0, 2), dtype=np.int64)
    edge = ply["edge"].data
    names = set(edge.dtype.names)
    if {"vertex1", "vertex2"}.issubset(names):
        return np.stack([edge["vertex1"], edge["vertex2"]], axis=1).astype(np.int64)
    if {"vertex_indices"}.issubset(names):
        return np.asarray([list(v)[:2] for v in edge["vertex_indices"]], dtype=np.int64)
    raise ValueError(f"Unsupported edge fields in MST ply: {edge.dtype.names}")


def _median_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1e-3
    dists, _ = cKDTree(points).query(points, k=2)
    return float(np.median(dists[:, 1].clip(min=1e-8)))


def _project_point_to_segment(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray):
    seg = seg_b - seg_a
    denom = np.dot(seg, seg) + 1e-12
    t = ((points - seg_a[None, :]) @ seg) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = seg_a[None, :] + t[:, None] * seg[None, :]
    dist = np.linalg.norm(points - proj, axis=1)
    return dist, t


def _dedupe_edges(edges, edge_types):
    merged = {}
    for (i, j), etype in zip(edges, edge_types):
        if i == j:
            continue
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if (a, b) not in merged:
            merged[(a, b)] = int(etype)
    out_edges = np.asarray(list(merged.keys()), dtype=np.int64)
    out_types = np.asarray(list(merged.values()), dtype=np.int64)
    return out_edges, out_types


def _match_subset_to_clean(
    clean_xyz: np.ndarray,
    subset_xyz: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(subset_xyz) == 0:
        return np.zeros((0,), dtype=np.int64)
    dists, idx = cKDTree(clean_xyz).query(subset_xyz, k=1)
    keep = dists <= threshold
    return np.unique(idx[keep].astype(np.int64))


def _assign_branch_leaf_labels(
    clean_xyz: np.ndarray,
    branch_xyz: np.ndarray,
    leaf_xyz: np.ndarray,
    threshold: float,
) -> np.ndarray:
    labels = np.full(len(clean_xyz), -1, dtype=np.int64)
    branch_idx = _match_subset_to_clean(clean_xyz, branch_xyz, threshold)
    leaf_idx = _match_subset_to_clean(clean_xyz, leaf_xyz, threshold)
    labels[branch_idx] = BRANCH_LABEL
    labels[leaf_idx] = LEAF_LABEL

    unlabeled = np.where(labels < 0)[0]
    if len(unlabeled) > 0:
        branch_tree = cKDTree(branch_xyz) if len(branch_xyz) > 0 else None
        leaf_tree = cKDTree(leaf_xyz) if len(leaf_xyz) > 0 else None
        for idx in unlabeled:
            pt = clean_xyz[idx]
            d_branch = branch_tree.query(pt, k=1)[0] if branch_tree is not None else np.inf
            d_leaf = leaf_tree.query(pt, k=1)[0] if leaf_tree is not None else np.inf
            labels[idx] = BRANCH_LABEL if d_branch <= d_leaf else LEAF_LABEL

    return labels


def _build_branch_edges(
    clean_xyz: np.ndarray,
    branch_idx: np.ndarray,
    mst_xyz: np.ndarray,
    mst_edges: np.ndarray,
    branch_support_knn: int,
):
    if len(branch_idx) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    branch_points = clean_xyz[branch_idx]
    branch_tree = cKDTree(branch_points)
    edges = []
    edge_types = []

    if len(mst_edges) == 0 or len(mst_xyz) == 0:
        k = min(max(branch_support_knn, 2), len(branch_points) - 1)
        if k >= 1:
            _, nn_idx = branch_tree.query(branch_points, k=k + 1)
            for local_i, nbrs in enumerate(nn_idx):
                for local_j in np.atleast_1d(nbrs)[1:]:
                    edges.append([branch_idx[local_i], branch_idx[int(local_j)]])
                    edge_types.append(EDGE_BRANCH_SUPPORT)
        return _dedupe_edges(edges, edge_types)

    seg_a = mst_xyz[mst_edges[:, 0]]
    seg_b = mst_xyz[mst_edges[:, 1]]
    seg_assign = np.zeros(len(branch_points), dtype=np.int64)
    seg_t = np.zeros(len(branch_points), dtype=np.float32)
    best_dist = np.full(len(branch_points), np.inf, dtype=np.float32)
    for seg_id, (a, b) in enumerate(zip(seg_a, seg_b)):
        dist, t = _project_point_to_segment(branch_points, a, b)
        update = dist < best_dist
        best_dist[update] = dist[update]
        seg_assign[update] = seg_id
        seg_t[update] = t[update]

    seg_to_locals = defaultdict(list)
    for local_idx, seg_id in enumerate(seg_assign.tolist()):
        seg_to_locals[int(seg_id)].append(local_idx)

    endpoint_to_branch = {}
    for node_id, node in enumerate(mst_xyz):
        local_idx = int(branch_tree.query(node, k=1)[1])
        endpoint_to_branch[node_id] = int(branch_idx[local_idx])

    for seg_id, local_ids in seg_to_locals.items():
        if not local_ids:
            continue
        ordered = sorted(local_ids, key=lambda i: float(seg_t[i]))
        ordered_global = [int(branch_idx[i]) for i in ordered]
        anchor_a = endpoint_to_branch[int(mst_edges[seg_id, 0])]
        anchor_b = endpoint_to_branch[int(mst_edges[seg_id, 1])]
        chain = [anchor_a] + ordered_global + [anchor_b]
        unique_chain = []
        for idx in chain:
            if not unique_chain or unique_chain[-1] != idx:
                unique_chain.append(idx)
        for i in range(len(unique_chain) - 1):
            edges.append([unique_chain[i], unique_chain[i + 1]])
            edge_types.append(EDGE_BRANCH_CHAIN)

        part_points = branch_points[np.asarray(local_ids, dtype=np.int64)]
        if len(part_points) > 2:
            k = min(branch_support_knn, len(part_points) - 1)
            _, nn_idx = cKDTree(part_points).query(part_points, k=k + 1)
            for local_i, nbrs in enumerate(nn_idx):
                for nbr in np.atleast_1d(nbrs)[1:]:
                    gi = int(branch_idx[local_ids[local_i]])
                    gj = int(branch_idx[local_ids[int(nbr)]])
                    edges.append([gi, gj])
                    edge_types.append(EDGE_BRANCH_SUPPORT)

    node_to_segments = defaultdict(list)
    for seg_id, (u, v) in enumerate(mst_edges.tolist()):
        node_to_segments[int(u)].append(seg_id)
        node_to_segments[int(v)].append(seg_id)
    for node_id, seg_ids in node_to_segments.items():
        if len(seg_ids) < 2:
            continue
        anchor = endpoint_to_branch[node_id]
        for seg_id in seg_ids:
            local_ids = seg_to_locals.get(seg_id, [])
            if not local_ids:
                continue
            pts = branch_points[np.asarray(local_ids, dtype=np.int64)]
            node = mst_xyz[node_id]
            nearest_local = local_ids[int(np.argmin(np.linalg.norm(pts - node[None, :], axis=1)))]
            edges.append([anchor, int(branch_idx[nearest_local])])
            edge_types.append(EDGE_BRANCH_SUPPORT)

    return _dedupe_edges(edges, edge_types)


def _connected_components(n_nodes: int, edges: np.ndarray):
    parent = np.arange(n_nodes, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges.tolist():
        union(int(i), int(j))

    groups = defaultdict(list)
    for i in range(n_nodes):
        groups[find(i)].append(i)
    return list(groups.values())


def _build_leaf_edges(
    clean_xyz: np.ndarray,
    branch_idx: np.ndarray,
    leaf_idx: np.ndarray,
    leaf_knn: int,
    leaf_radius_scale: float,
    leaf_attach_count: int,
):
    if len(leaf_idx) == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    leaf_points = clean_xyz[leaf_idx]
    branch_tree = cKDTree(clean_xyz[branch_idx]) if len(branch_idx) > 0 else None
    spacing = _median_spacing(leaf_points)
    radius = max(spacing * leaf_radius_scale, spacing * 1.5)
    leaf_tree = cKDTree(leaf_points)

    # Build connected leaf components once, then connect each component to the branch graph.
    local_pairs = np.asarray(list(leaf_tree.query_pairs(radius)), dtype=np.int64)
    if local_pairs.size == 0:
        components = [[i] for i in range(len(leaf_idx))]
    else:
        components = _connected_components(len(leaf_idx), local_pairs)

    edges = []
    edge_types = []

    for comp in components:
        comp_local = np.asarray(comp, dtype=np.int64)
        comp_global = leaf_idx[comp_local]
        comp_points = clean_xyz[comp_global]

        # One or a few attachment edges anchor the whole leaf component to the branch graph.
        if branch_tree is not None:
            dists, nbr_idx = branch_tree.query(comp_points, k=1)
            order = np.argsort(dists)
            keep = order[: max(1, min(leaf_attach_count, len(order)))]
            for k in keep.tolist():
                edges.append([int(comp_global[k]), int(branch_idx[int(nbr_idx[k])])])
                edge_types.append(EDGE_LEAF_ATTACH)

        # Keep leaf coherence with a sparse MST inside each attachment component, not dense KNN.
        if len(comp_local) > 1:
            k = min(max(int(leaf_knn), 1), len(comp_local) - 1)
            dists, nn_idx = cKDTree(comp_points).query(comp_points, k=k + 1)
            rows = []
            cols = []
            vals = []
            for row_id, (row_d, row_j) in enumerate(zip(dists, nn_idx)):
                for dist, nbr in zip(np.atleast_1d(row_d)[1:], np.atleast_1d(row_j)[1:]):
                    if not np.isfinite(dist):
                        continue
                    rows.append(int(row_id))
                    cols.append(int(nbr))
                    vals.append(float(dist))
                    rows.append(int(nbr))
                    cols.append(int(row_id))
                    vals.append(float(dist))
            graph = csr_matrix((vals, (rows, cols)), shape=(len(comp_local), len(comp_local)))
            mst = minimum_spanning_tree(graph).tocoo()
            for i, j in zip(mst.row.tolist(), mst.col.tolist()):
                edges.append([int(comp_global[i]), int(comp_global[j])])
                edge_types.append(EDGE_LEAF_INTRA)

    return _dedupe_edges(edges, edge_types)


def _make_spring_y(points: np.ndarray, edges: np.ndarray, edge_types: np.ndarray, stiffness: dict):
    rest = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).clip(min=1e-6)
    base = np.zeros(len(edges), dtype=np.float32)
    base[edge_types == EDGE_BRANCH_CHAIN] = stiffness["branch_chain"]
    base[edge_types == EDGE_BRANCH_SUPPORT] = stiffness["branch_support"]
    base[edge_types == EDGE_LEAF_INTRA] = stiffness["leaf_intra"]
    base[edge_types == EDGE_LEAF_ATTACH] = stiffness["leaf_attach"]
    ref = np.median(rest)
    scale = np.clip(np.sqrt(ref / rest), 0.7, 1.6).astype(np.float32)
    return (base * scale).astype(np.float32)


def _pose_to_camera(pose_json_path: str, width: int, height: int):
    if pose_json_path is None or not os.path.isfile(pose_json_path):
        intrinsic = np.array(
            [[width, 0.0, width / 2.0], [0.0, height, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        w2c = np.eye(4, dtype=np.float32)
        return intrinsic, w2c
    with open(pose_json_path, "r") as f:
        pose = json.load(f)
    if "pose" in pose:
        pose = pose["pose"][0]
    intrinsic = np.array(pose["intrinsic"], dtype=np.float32)
    intrinsic[0, :] *= float(width)
    intrinsic[1, :] *= float(height)
    w2c = np.array(pose["extrinsic"], dtype=np.float32)
    return intrinsic, w2c


def _camera_intrinsic_from_colmap(cam) -> np.ndarray:
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params[:4]
    elif cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = cam.params[:3]
        fx = fy = f
    else:
        raise ValueError(f"Unsupported COLMAP camera model for export: {cam.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _camera_from_colmap_sparse(sparse_dir: str, image_name: Optional[str] = None):
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    if not images:
        raise ValueError(f"No registered images found in {sparse_dir}")
    if image_name is None:
        chosen = images[sorted(images.keys())[0]]
    else:
        matched = [img for img in images.values() if img.name == image_name]
        if not matched:
            raise KeyError(f"{image_name} not found in {sparse_dir}/images.bin")
        chosen = matched[0]
    cam = cameras[chosen.camera_id]
    intrinsic = _camera_intrinsic_from_colmap(cam)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = chosen.qvec2rotmat().astype(np.float32)
    w2c[:3, 3] = chosen.tvec.astype(np.float32)
    return intrinsic, w2c, int(cam.width), int(cam.height), chosen.name


def main():
    parser = argparse.ArgumentParser(
        description="Export a plant Gaussian point cloud into an interaction-ready physics checkpoint and minimal data package."
    )
    parser.add_argument("--clean_ply", required=True)
    parser.add_argument("--branch_ply", required=True)
    parser.add_argument("--leaf_ply", required=True)
    parser.add_argument("--mst_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_name", default="plant_gaussian")
    parser.add_argument("--pose_json", default=None)
    parser.add_argument("--colmap_sparse_dir", default=None)
    parser.add_argument("--colmap_image_name", default=None)
    parser.add_argument(
        "--topology_mode",
        choices=["sampled_knn", "plant_attachment"],
        default="sampled_knn",
    )
    parser.add_argument("--physics_nodes", type=int, default=2000)
    parser.add_argument("--physics_knn", type=int, default=8)
    parser.add_argument("--physics_seed", type=int, default=0)
    parser.add_argument("--bg_img_path", default="data/bg.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--label_match_scale", type=float, default=3.0)
    parser.add_argument("--branch_support_knn", type=int, default=3)
    parser.add_argument("--leaf_knn", type=int, default=4)
    parser.add_argument("--leaf_radius_scale", type=float, default=3.0)
    parser.add_argument("--leaf_attach_count", type=int, default=2)
    parser.add_argument("--branch_chain_stiffness", type=float, default=30000.0)
    parser.add_argument("--branch_support_stiffness", type=float, default=18000.0)
    parser.add_argument("--leaf_intra_stiffness", type=float, default=1200.0)
    parser.add_argument("--leaf_attach_stiffness", type=float, default=4500.0)
    parser.add_argument("--collision_dist", type=float, default=0.02)
    parser.add_argument("--dashpot_damping", type=float, default=80.0)
    parser.add_argument("--drag_damping", type=float, default=2.0)
    parser.add_argument("--collide_elas", type=float, default=0.5)
    parser.add_argument("--collide_fric", type=float, default=0.3)
    parser.add_argument("--collide_object_elas", type=float, default=0.7)
    parser.add_argument("--collide_object_fric", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clean_xyz, clean_rgb, _ = _read_ply_xyz_rgb(args.clean_ply)
    branch_xyz, _, _ = _read_ply_xyz_rgb(args.branch_ply)
    leaf_xyz, _, _ = _read_ply_xyz_rgb(args.leaf_ply)
    mst_xyz, _, mst_ply = _read_ply_xyz_rgb(args.mst_ply)
    mst_edges = _read_ply_edges(mst_ply)

    clean_spacing = _median_spacing(clean_xyz)
    label_threshold = max(clean_spacing * args.label_match_scale, 1e-4)
    full_labels = _assign_branch_leaf_labels(
        clean_xyz, branch_xyz, leaf_xyz, label_threshold
    )

    if args.topology_mode == "sampled_knn":
        sampled_idx = _farthest_point_sample_indices(
            clean_xyz, args.physics_nodes, seed=args.physics_seed
        )
        points_xyz = clean_xyz[sampled_idx]
        point_labels = full_labels[sampled_idx]
        point_rgb = clean_rgb[sampled_idx] if clean_rgb is not None else None
        branch_idx = np.where(point_labels == BRANCH_LABEL)[0]
        leaf_idx = np.where(point_labels == LEAF_LABEL)[0]
        edges, edge_types = _build_sampled_knn_topology(
            points_xyz,
            point_labels,
            knn=args.physics_knn,
        )
    else:
        sampled_idx = np.arange(len(clean_xyz), dtype=np.int64)
        points_xyz = clean_xyz
        point_labels = full_labels
        point_rgb = clean_rgb
        branch_idx = np.where(point_labels == BRANCH_LABEL)[0]
        leaf_idx = np.where(point_labels == LEAF_LABEL)[0]

        branch_edges, branch_edge_types = _build_branch_edges(
            points_xyz,
            branch_idx,
            mst_xyz,
            mst_edges,
            branch_support_knn=args.branch_support_knn,
        )
        leaf_edges, leaf_edge_types = _build_leaf_edges(
            points_xyz,
            branch_idx,
            leaf_idx,
            leaf_knn=args.leaf_knn,
            leaf_radius_scale=args.leaf_radius_scale,
            leaf_attach_count=args.leaf_attach_count,
        )

        edges = np.concatenate([branch_edges, leaf_edges], axis=0)
        edge_types = np.concatenate([branch_edge_types, leaf_edge_types], axis=0)
        edges, edge_types = _dedupe_edges(edges, edge_types)
    if len(edges) == 0:
        raise ValueError("No topology edges were built for the plant.")

    spring_y = _make_spring_y(
        points_xyz,
        edges,
        edge_types,
        stiffness={
            "branch_chain": args.branch_chain_stiffness,
            "branch_support": args.branch_support_stiffness,
            "leaf_intra": args.leaf_intra_stiffness,
            "leaf_attach": args.leaf_attach_stiffness,
        },
    )

    if point_rgb is None:
        point_rgb = np.zeros((len(points_xyz), 3), dtype=np.float32)
        point_rgb[point_labels == BRANCH_LABEL] = np.array([0.45, 0.28, 0.12], dtype=np.float32)
        point_rgb[point_labels == LEAF_LABEL] = np.array([0.18, 0.62, 0.22], dtype=np.float32)

    physics_path = os.path.join(args.output_dir, f"{args.case_name}_physics.pth")
    data_path = os.path.join(args.output_dir, f"{args.case_name}_final_data.pkl")
    metadata_path = os.path.join(args.output_dir, f"{args.case_name}_metadata.json")
    calibrate_path = os.path.join(args.output_dir, f"{args.case_name}_calibrate.pkl")
    debug_npz_path = os.path.join(args.output_dir, f"{args.case_name}_topology_debug.npz")
    manifest_path = os.path.join(args.output_dir, f"{args.case_name}_interaction_manifest.json")

    torch.save(
        {
            "spring_Y": torch.from_numpy(spring_y),
            "collide_elas": torch.tensor([args.collide_elas], dtype=torch.float32),
            "collide_fric": torch.tensor([args.collide_fric], dtype=torch.float32),
            "collide_object_elas": torch.tensor([args.collide_object_elas], dtype=torch.float32),
            "collide_object_fric": torch.tensor([args.collide_object_fric], dtype=torch.float32),
            "collision_dist": torch.tensor([args.collision_dist], dtype=torch.float32),
            "dashpot_damping": torch.tensor([args.dashpot_damping], dtype=torch.float32),
            "drag_damping": torch.tensor([args.drag_damping], dtype=torch.float32),
            "num_object_springs": torch.tensor(len(edges), dtype=torch.long),
            "edges": torch.from_numpy(edges),
            "edge_mid": torch.from_numpy(((points_xyz[edges[:, 0]] + points_xyz[edges[:, 1]]) * 0.5).astype(np.float32)),
            "point_labels": torch.from_numpy(point_labels),
            "edge_types": torch.from_numpy(edge_types),
            "edge_type_names": {
                EDGE_BRANCH_CHAIN: "branch_chain",
                EDGE_BRANCH_SUPPORT: "branch_support",
                EDGE_LEAF_INTRA: "leaf_intra",
                EDGE_LEAF_ATTACH: "leaf_attach",
            },
            "topology_cfg": {
                "mode": args.topology_mode,
                "label_match_threshold": float(label_threshold),
                "branch_support_knn": int(args.branch_support_knn),
                "leaf_knn": int(args.leaf_knn),
                "leaf_radius_scale": float(args.leaf_radius_scale),
                "leaf_attach_count": int(args.leaf_attach_count),
                "physics_nodes": int(len(points_xyz)),
                "physics_knn": int(args.physics_knn),
            },
            "sampled_idx": torch.from_numpy(sampled_idx),
            "source_paths": {
                "clean_ply": args.clean_ply,
                "branch_ply": args.branch_ply,
                "leaf_ply": args.leaf_ply,
                "mst_ply": args.mst_ply,
                "pose_json": args.pose_json,
                "colmap_sparse_dir": args.colmap_sparse_dir,
                "colmap_image_name": args.colmap_image_name,
            },
        },
        physics_path,
    )

    final_data = {
        "object_points": points_xyz[None].astype(np.float32),
        "object_colors": point_rgb[None].astype(np.float32),
        "object_visibilities": np.ones((1, len(points_xyz)), dtype=bool),
        "object_motions_valid": np.ones((1, len(points_xyz)), dtype=bool),
        "controller_points": np.zeros((1, 0, 3), dtype=np.float32),
        "surface_points": np.zeros((0, 3), dtype=np.float32),
        "interior_points": np.zeros((0, 3), dtype=np.float32),
    }
    with open(data_path, "wb") as f:
        pickle.dump(final_data, f)

    source_view = None
    if args.colmap_sparse_dir is not None:
        intrinsic, w2c, args.width, args.height, source_view = _camera_from_colmap_sparse(
            args.colmap_sparse_dir,
            image_name=args.colmap_image_name,
        )
    else:
        intrinsic, w2c = _pose_to_camera(args.pose_json, args.width, args.height)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "intrinsics": [intrinsic.tolist()],
                "WH": [args.width, args.height],
                "bg_img_path": args.bg_img_path,
                "source_view": source_view,
            },
            f,
            indent=2,
        )
    with open(calibrate_path, "wb") as f:
        pickle.dump([np.linalg.inv(w2c).astype(np.float32)], f)

    np.savez_compressed(
        debug_npz_path,
        clean_xyz=clean_xyz,
        labels=full_labels,
        sampled_xyz=points_xyz,
        sampled_idx=sampled_idx,
        sampled_labels=point_labels,
        edges=edges,
        edge_types=edge_types,
        spring_y=spring_y,
        branch_idx=branch_idx,
        leaf_idx=leaf_idx,
        mst_xyz=mst_xyz,
        mst_edges=mst_edges,
    )

    manifest = {
        "case_name": args.case_name,
        "physics_path": physics_path,
        "data_path": data_path,
        "metadata_path": metadata_path,
        "calibrate_path": calibrate_path,
        "debug_npz_path": debug_npz_path,
        "pose_json": args.pose_json,
        "colmap_sparse_dir": args.colmap_sparse_dir,
        "colmap_image_name": args.colmap_image_name,
        "source_view": source_view,
        "bg_img_path": args.bg_img_path,
        "width": args.width,
        "height": args.height,
        "stats": {
            "n_full_points": int(len(clean_xyz)),
            "n_physics_points": int(len(points_xyz)),
            "n_branch_points": int(len(branch_idx)),
            "n_leaf_points": int(len(leaf_idx)),
            "n_edges": int(len(edges)),
            "spring_min": float(spring_y.min()),
            "spring_max": float(spring_y.max()),
            "spring_mean": float(spring_y.mean()),
        },
        "launch_example": [
            "python",
            "semantic/run_direct_object_drag_playground.py",
            "--data_path",
            data_path,
            "--physics_path",
            physics_path,
            "--gs_path",
            args.clean_ply,
            "--bg_img_path",
            args.bg_img_path,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ],
    }
    if args.pose_json is not None:
        manifest["launch_example"].extend(["--pose_json", args.pose_json])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] physics: {physics_path}")
    print(f"[done] final_data: {data_path}")
    print(
        f"[info] points={len(clean_xyz)} branch={len(branch_idx)} leaf={len(leaf_idx)} "
        f"edges={len(edges)} spring_mean={spring_y.mean():.2f}"
    )


if __name__ == "__main__":
    main()
