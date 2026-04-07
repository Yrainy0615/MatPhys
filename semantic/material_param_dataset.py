import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset



def _build_object_edges_open3d(
    points: np.ndarray,
    use_knn_topology: bool,
    object_knn: int,
    object_radius: float,
    object_max_neighbours: int,
) -> np.ndarray:
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

    spring_flags = np.zeros((len(points), len(points)), dtype=np.uint8)
    springs: List[List[int]] = []
    for i in range(len(points)):
        if use_knn_topology:
            _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], object_knn + 1)
        else:
            _, idx, _ = pcd_tree.search_hybrid_vector_3d(
                points[i], object_radius, object_max_neighbours
            )
        idx = idx[1:]
        for j in idx:
            rest_length = np.linalg.norm(points[i] - points[j])
            if (
                spring_flags[i, j] == 0
                and spring_flags[j, i] == 0
                and rest_length > 1e-4
            ):
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                springs.append([i, j])
    return np.asarray(springs, dtype=np.int64)


def _geo_features(
    points: np.ndarray,
    edges: np.ndarray,
    point_part: np.ndarray,
    edge_part_idx: np.ndarray,
    density_k: int = 16,
) -> np.ndarray:
    """Part-aware, position-invariant geometry features per edge.

    Edge-level features:
        l0_norm   [1]  normalized rest length (relative to part median)
        abs_dir   [3]  absolute direction (order-invariant)

    Intra-part node features (broadcast to edges, symmetric):
        deg_min   [1]  min intra-part degree (normalized by part mean)
        deg_max   [1]  max intra-part degree
        dens_mean [1]  mean intra-part local density
        dens_diff [1]  intra-part density contrast

    Part-level PCA features (broadcast to edges):
        pca_ratio1 [1]  λ2/λ1 (elongation)
        pca_ratio2 [1]  λ3/λ1 (flatness)
        pca_spread [1]  log(λ1) normalized spread

    Total: 11 dimensions
    """
    n_points = points.shape[0]
    e_i, e_j = edges[:, 0], edges[:, 1]
    p_i, p_j = points[e_i], points[e_j]
    num_parts = int(point_part.max()) + 1

    # === Edge-level: rest length & direction ===
    vec = p_j - p_i
    l0 = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-8)
    direction = vec / l0
    abs_dir = np.abs(direction)  # [E, 3]

    # Normalize rest length per-part (relative to part median)
    l0_norm = np.zeros_like(l0)
    for p in range(num_parts):
        mask = edge_part_idx == p
        if mask.sum() == 0:
            continue
        part_median = np.median(l0[mask])
        l0_norm[mask] = l0[mask] / (part_median + 1e-8)

    # === Intra-part node degree ===
    # Only count edges within the same part for each node
    part_degree = np.zeros(n_points, dtype=np.float32)
    for idx in range(len(edges)):
        i, j = edges[idx, 0], edges[idx, 1]
        # Count if both endpoints are in the same part
        if point_part[i] == point_part[j]:
            part_degree[i] += 1
            part_degree[j] += 1

    # Normalize degree per part
    part_deg_norm = np.zeros(n_points, dtype=np.float32)
    for p in range(num_parts):
        mask = point_part == p
        if mask.sum() == 0:
            continue
        pmean = part_degree[mask].mean() + 1e-8
        part_deg_norm[mask] = part_degree[mask] / pmean

    deg_i = part_deg_norm[e_i]
    deg_j = part_deg_norm[e_j]
    deg_min = np.minimum(deg_i, deg_j)[:, None]  # [E, 1]
    deg_max = np.maximum(deg_i, deg_j)[:, None]  # [E, 1]

    # === Intra-part local density ===
    local_density = np.zeros(n_points, dtype=np.float32)
    for p in range(num_parts):
        mask = point_part == p
        part_pts = points[mask]
        n_part = part_pts.shape[0]
        if n_part < 2:
            local_density[mask] = 1.0
            continue
        # NN distances within part
        d2 = np.sum((part_pts[:, None, :] - part_pts[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        k = min(density_k, n_part - 1)
        nn_dists = np.partition(d2, kth=k - 1, axis=1)[:, :k]
        dens = 1.0 / (np.sqrt(nn_dists.mean(axis=1)) + 1e-8)
        # Normalize within part
        dens_med = np.median(dens) + 1e-8
        local_density[mask] = dens / dens_med

    dens_i = local_density[e_i]
    dens_j = local_density[e_j]
    dens_mean = ((dens_i + dens_j) / 2)[:, None]  # [E, 1]
    dens_diff = np.abs(dens_i - dens_j)[:, None]   # [E, 1]

    # === Part-level PCA (shape descriptors) ===
    # Per-part: PCA eigenvalue ratios → broadcast to edges
    pca_ratio1 = np.zeros(num_parts, dtype=np.float32)  # λ2/λ1
    pca_ratio2 = np.zeros(num_parts, dtype=np.float32)  # λ3/λ1
    pca_spread = np.zeros(num_parts, dtype=np.float32)  # log(λ1)
    for p in range(num_parts):
        mask = point_part == p
        part_pts = points[mask]
        if part_pts.shape[0] < 3:
            pca_ratio1[p] = 1.0
            pca_ratio2[p] = 1.0
            pca_spread[p] = 0.0
            continue
        centered = part_pts - part_pts.mean(axis=0)
        cov = (centered.T @ centered) / (part_pts.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)  # sorted ascending
        eigvals = eigvals[::-1].clip(min=1e-12)  # descending
        pca_ratio1[p] = eigvals[1] / eigvals[0]
        pca_ratio2[p] = eigvals[2] / eigvals[0]
        pca_spread[p] = np.log(eigvals[0] + 1e-8)

    # Normalize spread across parts
    sp_med = np.median(np.abs(pca_spread[pca_spread != 0])) + 1e-8
    pca_spread = pca_spread / sp_med

    # Broadcast to edges
    edge_pca1 = pca_ratio1[edge_part_idx][:, None]   # [E, 1]
    edge_pca2 = pca_ratio2[edge_part_idx][:, None]   # [E, 1]
    edge_spread = pca_spread[edge_part_idx][:, None]  # [E, 1]

    # === Assemble ===
    feat = np.concatenate([
        l0_norm,      # [1] rest length (per-part normalized)
        abs_dir,      # [3] absolute direction
        deg_min,      # [1] intra-part degree (min)
        deg_max,      # [1] intra-part degree (max)
        dens_mean,    # [1] intra-part density (mean)
        dens_diff,    # [1] intra-part density contrast
        edge_pca1,    # [1] PCA ratio λ2/λ1 (elongation)
        edge_pca2,    # [1] PCA ratio λ3/λ1 (flatness)
        edge_spread,  # [1] PCA log(λ1) (spread)
    ], axis=1)

    return feat.astype(np.float32)  # [E, 11]


def _edge_sem_from_node(node_sem: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return (node_sem[edges[:, 0]] + node_sem[edges[:, 1]]) * 0.5


def _build_controller_springs(
    structure_points: np.ndarray,
    controller_points_frame0: np.ndarray,
    topology_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build controller-object springs (same logic as trainer_warp._init_start).

    Returns:
        ctrl_obj_idx: [C] object point index for each controller spring
        ctrl_rest_length: [C, 1] rest length per controller spring
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(structure_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    n_obj = len(structure_points)

    obj_indices = []
    rest_lengths = []
    use_knn = topology_cfg.get("use_knn_topology", False)
    ctrl_knn = topology_cfg.get("controller_knn", 10)
    ctrl_radius = topology_cfg.get("controller_radius", 0.05)
    ctrl_max = topology_cfg.get("controller_max_neighbours", 20)

    for i in range(len(controller_points_frame0)):
        pt = controller_points_frame0[i]
        if use_knn:
            _, idx, _ = pcd_tree.search_knn_vector_3d(pt, min(ctrl_knn, n_obj))
        else:
            _, idx, _ = pcd_tree.search_hybrid_vector_3d(pt, ctrl_radius, ctrl_max)
        for j in idx:
            obj_indices.append(j)
            rest_lengths.append(np.linalg.norm(pt - structure_points[j]))

    if len(obj_indices) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32)

    return (
        np.array(obj_indices, dtype=np.int64),
        np.array(rest_lengths, dtype=np.float32).reshape(-1, 1),
    )


def _edge_midpoints(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return ((points[edges[:, 0]] + points[edges[:, 1]]) * 0.5).astype(np.float32)


def _pick_best_ckpt(train_dir: str) -> str:
    best_list = sorted(glob.glob(os.path.join(train_dir, "best_*.pth")))
    if not best_list:
        raise FileNotFoundError(f"No best_*.pth found in {train_dir}")
    return best_list[-1]


def _load_optimal_params(experiments_optimization_dir: str, case_name: str) -> Dict:
    optimal_path = os.path.join(
        experiments_optimization_dir, case_name, "optimal_params.pkl"
    )
    if not os.path.exists(optimal_path):
        return {}
    with open(optimal_path, "rb") as f:
        return pickle.load(f)

    
@dataclass
class MaterialDatasetConfig:
    base_path: str
    sem_cache_dir: str
    experiments_dir: str
    experiments_optimization_dir: str
    case_to_material_path: str
    results_dir: str = "results"  # Where extract_material_parts.py outputs are stored
    use_knn_topology: bool = False
    object_knn: int = 20
    object_radius: float = 0.02
    object_max_neighbours: int = 30


class MaterialParamDataset(Dataset):
    def __init__(self, cfg: MaterialDatasetConfig):
        self.cfg = cfg
        self.case_to_material, self.class_to_id = self._load_case_to_material(
            cfg.case_to_material_path
        )
        self.case_names = sorted(self.case_to_material.keys())
        self.samples: List[Dict[str, torch.Tensor]] = []
        for case_name in self.case_names:
            self.samples.append(self._build_sample(case_name))

    @staticmethod
    def _load_case_to_material(mapping_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        with open(mapping_path, "r") as f:
            raw = json.load(f)
        if "case_to_material" in raw:
            case_to_material = raw["case_to_material"]
            class_to_id = raw.get("class_to_id", {})
        else:
            case_to_material = raw
            class_to_id = {}

        example_value = next(iter(case_to_material.values()))
        if isinstance(example_value, str):
            if not class_to_id:
                class_names = sorted(set(case_to_material.values()))
                class_to_id = {name: idx for idx, name in enumerate(class_names)}
            case_to_material_id = {
                case: int(class_to_id[label]) for case, label in case_to_material.items()
            }
        else:
            case_to_material_id = {
                case: int(label) for case, label in case_to_material.items()
            }
            if not class_to_id:
                class_to_id = {
                    str(idx): idx for idx in sorted(set(case_to_material_id.values()))
                }
        return case_to_material_id, class_to_id

    def _build_sample(self, case_name: str) -> Dict[str, torch.Tensor]:
        case_dir = os.path.join(self.cfg.base_path, case_name)
        final_data_path = os.path.join(case_dir, "final_data.pkl")
        with open(final_data_path, "rb") as f:
            final_data = pickle.load(f)

        object_points = final_data["object_points"][0]
        other_surface_points = final_data["surface_points"]
        interior_points = final_data["interior_points"]
        points = np.concatenate(
            [object_points, other_surface_points, interior_points], axis=0
        ).astype(np.float32)

        topology_cfg = self._load_topology_from_optimization(case_name)
        edges = _build_object_edges_open3d(
            points=points,
            use_knn_topology=topology_cfg["use_knn_topology"],
            object_knn=topology_cfg["object_knn"],
            object_radius=topology_cfg["object_radius"],
            object_max_neighbours=topology_cfg["object_max_neighbours"],
        )
        edge_mid = _edge_midpoints(points, edges)

        sem_path = os.path.join(self.cfg.sem_cache_dir, f"{case_name}_node_sem.npz")
        node_sem = np.load(sem_path)["node_sem"].astype(np.float32)
        if node_sem.shape[0] != points.shape[0]:
            raise ValueError(
                f"node_sem point count mismatch for {case_name}: "
                f"{node_sem.shape[0]} vs {points.shape[0]}"
            )
        z_sem = _edge_sem_from_node(node_sem, edges)
        z_sem_global = node_sem.mean(axis=0, keepdims=True).astype(np.float32)  # [1, D]

        # Load VLM-precomputed material distribution and part assignments
        material_dist, edge_part_idx, point_part, part_features = self._load_material_dist(
            case_name, points, edges
        )

        # Compute part-aware geometry features (needs part assignments)
        point_part_np = point_part.numpy()
        edge_part_idx_np = edge_part_idx.numpy()
        z_geo = _geo_features(points, edges, point_part_np, edge_part_idx_np)

        ckpt = torch.load(
            _pick_best_ckpt(os.path.join(self.cfg.experiments_dir, case_name, "train")),
            map_location="cpu",
            weights_only=False,
        )
        num_object_springs = int(ckpt.get("num_object_springs", ckpt["spring_Y"].numel()))
        teacher_k = ckpt["spring_Y"].float().view(-1, 1)[:num_object_springs]
        base_spring_y = ckpt["spring_Y"].float().view(-1)
        if teacher_k.shape[0] != edges.shape[0]:
            raise ValueError(
                f"teacher_k size mismatch for {case_name}: {teacher_k.shape[0]} vs {edges.shape[0]} "
                f"(topology mismatch with first-order run; "
                f"use_knn_topology={topology_cfg['use_knn_topology']}, "
                f"object_knn={topology_cfg['object_knn']}, "
                f"object_radius={topology_cfg['object_radius']}, "
                f"object_max_neighbours={topology_cfg['object_max_neighbours']})"
            )

        # Build controller spring features
        controller_points = final_data.get("controller_points", None)
        if controller_points is not None:
            ctrl_obj_idx, ctrl_rest_length = _build_controller_springs(
                points, controller_points[0], topology_cfg,
            )
            # Semantic features of the object endpoint
            ctrl_sem = node_sem[ctrl_obj_idx]                    # [C, D]
            # Part index of the object endpoint
            ctrl_part_idx = point_part[ctrl_obj_idx]             # [C]
        else:
            ctrl_sem = np.zeros((0, node_sem.shape[1]), dtype=np.float32)
            ctrl_rest_length = np.zeros((0, 1), dtype=np.float32)
            ctrl_part_idx = torch.zeros(0, dtype=torch.long)

        split_path = os.path.join(case_dir, "split.json")
        with open(split_path, "r") as f:
            split = json.load(f)
        train_frame = int(split["train"][1])

        optimal_params = _load_optimal_params(
            self.cfg.experiments_optimization_dir, case_name
        )
        teacher_collision_dist = float(optimal_params.get("collision_dist", 0.02))
        teacher_drag_damping = float(optimal_params.get("drag_damping", 3.0))
        teacher_dashpot_damping = float(optimal_params.get("dashpot_damping", 100.0))

        metadata_path = os.path.join(case_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
        with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
            c2ws = np.asarray(pickle.load(f), dtype=np.float32)
        w2cs = np.linalg.inv(c2ws).astype(np.float32)
        wh = np.asarray(metadata["WH"], dtype=np.int64)

        return {
            "case_name": case_name,
            "case_dir": case_dir,
            "final_data_path": final_data_path,
            "train_frame": torch.tensor(train_frame, dtype=torch.long),
            "cam0_intrinsics": torch.from_numpy(intrinsics[0]).float(),
            "cam0_w2c": torch.from_numpy(w2cs[0]).float(),
            "wh": torch.from_numpy(wh).long(),
            "edges": torch.from_numpy(edges).long(),
            "z_geo": torch.from_numpy(z_geo).float(),
            "edge_mid": torch.from_numpy(edge_mid).float(),
            "z_sem": torch.from_numpy(z_sem).float(),
            "z_sem_global": torch.from_numpy(z_sem_global).float(),
            "part_features": part_features.float(),      # [K, D]
            "material_dist": material_dist,        # [K, M]
            "edge_part_idx": edge_part_idx,        # [E]
            # Controller spring features
            "ctrl_sem": torch.from_numpy(ctrl_sem).float(),            # [C, D]
            "ctrl_rest_length": torch.from_numpy(ctrl_rest_length).float(),  # [C, 1]
            "ctrl_part_idx": ctrl_part_idx if isinstance(ctrl_part_idx, torch.Tensor)
                             else torch.from_numpy(ctrl_part_idx).long(),    # [C]
            "num_object_springs": torch.tensor(num_object_springs, dtype=torch.long),
            "teacher_logk": teacher_k.log(),
            "base_spring_y": base_spring_y,
            "collide_elas": ckpt["collide_elas"].float().view(1),
            "collide_fric": ckpt["collide_fric"].float().view(1),
            "collide_object_elas": ckpt["collide_object_elas"].float().view(1),
            "collide_object_fric": ckpt["collide_object_fric"].float().view(1),
            "teacher_collision_dist": torch.tensor(
                [teacher_collision_dist], dtype=torch.float32
            ),
            "teacher_drag_damping": torch.tensor(
                [teacher_drag_damping], dtype=torch.float32
            ),
            "teacher_dashpot_damping": torch.tensor(
                [teacher_dashpot_damping], dtype=torch.float32
            ),
            "material_id": torch.tensor(int(self.case_to_material[case_name]), dtype=torch.long),
        }

    def _load_material_dist(
        self, case_name: str, points: np.ndarray, edges: np.ndarray,
    ) -> tuple:
        """Load VLM-precomputed material distribution and part assignments.

        From extract_material_parts.py output:
        - part_assignments [N_gs]: per-gaussian hard part index
        - material_distributions [K, M]: per-part material probs from VLM

        We transfer gaussian part assignments to structure points via NN,
        then assign each edge to a part.

        Returns:
            material_dist: [K, M] per-part material distribution
            edge_part_idx: [E] part index per edge
            point_part: [N_pts] per-node part index (for controller springs)
        """
        train_ready_path = os.path.join(
            self.cfg.results_dir, case_name, "train", "train_ready.pt"
        )
        N = points.shape[0]
        E = edges.shape[0]

        if not os.path.exists(train_ready_path):
            K, M = 1, 10
            print(f"[warn] {case_name}: train_ready.pt not found, using uniform fallback")
            return (
                torch.ones(K, M, dtype=torch.float32) / M,
                torch.zeros(E, dtype=torch.long),
                torch.zeros(N, dtype=torch.long),
                torch.zeros((K, 1024), dtype=torch.float32),
            )

        train_ready = torch.load(train_ready_path, map_location="cpu", weights_only=False)
        part_assignments = train_ready["part_assignments"]      # [N_gs]
        mat_dists = train_ready["material_distributions"]       # [K, M]
        part_features = train_ready.get("part_features", None)  # [K, D]
        gs_xyz = train_ready["xyz"].numpy()                     # [N_gs, 3]

        # Transfer gaussian part assignments to structure points via NN
        from scipy.spatial import cKDTree
        tree = cKDTree(gs_xyz)
        _, nn_idx = tree.query(points, k=1)  # [N_pts]
        point_part = part_assignments[nn_idx]  # [N_pts]

        # Clamp unassigned (-1) to 0
        point_part = point_part.clamp(min=0)

        # Per-edge: use first endpoint's part (both endpoints are usually in the same part)
        edge_part_idx = point_part[edges[:, 0]]  # [E]

        self.num_materials = int(mat_dists.shape[1])

        if part_features is None:
            D = 1024
            K = int(mat_dists.shape[0])
            part_features = torch.zeros((K, D), dtype=torch.float32)

        return (
            mat_dists.float(),
            edge_part_idx.long(),
            point_part.long(),
            part_features.float(),
        )

    def _load_topology_from_optimization(self, case_name: str) -> Dict[str, float]:
        cfg = {
            "use_knn_topology": bool(self.cfg.use_knn_topology),
            "object_knn": int(self.cfg.object_knn),
            "object_radius": float(self.cfg.object_radius),
            "object_max_neighbours": int(self.cfg.object_max_neighbours),
            # Controller spring topology defaults
            "controller_radius": 0.04,
            "controller_max_neighbours": 50,
            "controller_knn": 30,
        }
        optimal_path = os.path.join(
            self.cfg.experiments_optimization_dir, case_name, "optimal_params.pkl"
        )
        if not os.path.exists(optimal_path):
            return cfg

        optimal_params = _load_optimal_params(
            self.cfg.experiments_optimization_dir, case_name
        )

        for key in cfg:
            if key in optimal_params:
                cfg[key] = type(cfg[key])(optimal_params[key])
        return cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    keys = [
        "case_name", "case_dir", "final_data_path", "train_frame",
        "cam0_intrinsics", "cam0_w2c", "wh", "edges",
        "z_geo", "edge_mid", "z_sem", "z_sem_global", "part_features", "material_dist", "edge_part_idx",
        "ctrl_sem", "ctrl_rest_length", "ctrl_part_idx", "num_object_springs",
        "teacher_logk", "base_spring_y",
        "collide_elas", "collide_fric", "collide_object_elas", "collide_object_fric",
        "teacher_collision_dist", "teacher_drag_damping", "teacher_dashpot_damping",
        "material_id",
    ]
    out: Dict[str, list] = {k: [] for k in keys}
    for sample in batch:
        for k in keys:
            if k in sample:
                out[k].append(sample[k])
    return out


def create_dataloader(
    cfg: MaterialDatasetConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> Tuple[MaterialParamDataset, DataLoader]:
    dataset = MaterialParamDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )
    return dataset, loader


def create_train_test_dataloaders(
    cfg: MaterialDatasetConfig,
    batch_size: int,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[MaterialParamDataset, DataLoader, DataLoader]:
    """Create train and test dataloaders with random 8:2 split (train_ratio : 1-train_ratio).

    Cases are shuffled with the given seed, then the first train_ratio fraction
    is used for training and the rest for testing.

    Args:
        cfg: Dataset configuration
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        train_ratio: Fraction of samples for training (default 0.8 = 8:2)
        seed: Random seed for reproducible split

    Returns:
        full_dataset: Full dataset
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    import random

    full_dataset = MaterialParamDataset(cfg)
    n_samples = len(full_dataset)

    # Random 8:2 split: shuffle case indices with seed
    indices = list(range(n_samples))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )

    train_cases = [full_dataset.samples[i]["case_name"] for i in train_indices]
    test_cases = [full_dataset.samples[i]["case_name"] for i in test_indices]
    print(f"[Dataset Split] Random 8:2 (seed={seed}) — Total: {n_samples}, Train: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"  Train: {train_cases}")
    print(f"  Test:  {test_cases}")

    return full_dataset, train_loader, test_loader
