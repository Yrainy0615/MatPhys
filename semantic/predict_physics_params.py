import argparse
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from material_param_dataset import _build_object_edges_open3d, _edge_midpoints, _geo_features
from models import EdgeLevelMaterialPhysics, PartLevelMaterialPhysics


def create_model_for_type(
    model_type: str,
    dino_dim: int = 1024,
    geo_input_dim: int = 10,
    num_materials: int = 10,
):
    if model_type == "edge_level":
        return EdgeLevelMaterialPhysics(
            sem_dim=dino_dim,
            dino_dim=dino_dim,
            geo_input_dim=geo_input_dim,
            num_materials=num_materials,
        )
    if model_type == "part_level":
        return PartLevelMaterialPhysics(num_materials=num_materials)
    raise ValueError(f"Unknown model_type: {model_type}")


def forward_model(
    model,
    model_type: str,
    z_geo: torch.Tensor,
    z_sem: torch.Tensor,
    material_dist: torch.Tensor,
    edge_part_idx: torch.Tensor,
    z_sem_global: torch.Tensor = None,
    ctrl_sem: torch.Tensor = None,
    ctrl_rest_length: torch.Tensor = None,
    ctrl_part_idx: torch.Tensor = None,
):
    if model_type == "edge_level":
        return model(
            z_geo,
            z_sem,
            material_dist,
            edge_part_idx,
            z_sem_global,
            ctrl_sem=ctrl_sem,
            ctrl_rest_length=ctrl_rest_length,
            ctrl_part_idx=ctrl_part_idx,
        )
    if model_type == "part_level":
        return {"log_k": model(material_dist, edge_part_idx)}
    raise ValueError(f"Unknown model_type: {model_type}")


def _load_array(path, key=None):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if key is not None:
            if key not in data:
                raise KeyError(f"{path}: key '{key}' not found. Available: {list(data.keys())}")
            return data[key]
        if len(data.files) != 1:
            raise ValueError(
                f"{path}: contains multiple arrays {list(data.files)}; pass --object_points_key"
            )
        return data[data.files[0]]
    if path.endswith(".ply"):
        ply = PlyData.read(path)
        v = ply["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=1)
    raise ValueError(f"Unsupported array file: {path}")


def build_features(points, train_ready, topology_cfg):
    edges = _build_object_edges_open3d(
        points=points,
        use_knn_topology=topology_cfg["use_knn_topology"],
        object_knn=topology_cfg["object_knn"],
        object_radius=topology_cfg["object_radius"],
        object_max_neighbours=topology_cfg["object_max_neighbours"],
    )
    if edges.shape[0] == 0:
        raise ValueError("No object edges were built. Check topology parameters.")

    part_assignments = train_ready["part_assignments"].long()
    material_dist = train_ready["material_distributions"].float()
    part_features = train_ready["part_features"].float()
    gs_xyz = train_ready["xyz"].detach().cpu().numpy().astype(np.float32)

    tree = cKDTree(gs_xyz)
    _, nn_idx = tree.query(points, k=1)
    point_part = part_assignments[nn_idx].clamp(min=0)
    point_sem = part_features[point_part]
    edge_part_idx = point_part[edges[:, 0]]
    z_sem = 0.5 * (point_sem[edges[:, 0]] + point_sem[edges[:, 1]])
    z_sem_global = point_sem.mean(dim=0, keepdim=True)

    z_geo = _geo_features(
        points,
        edges,
        point_part.detach().cpu().numpy(),
        edge_part_idx.detach().cpu().numpy(),
    )

    return {
        "edges": torch.from_numpy(edges).long(),
        "edge_mid": torch.from_numpy(_edge_midpoints(points, edges)).float(),
        "z_geo": torch.from_numpy(z_geo).float(),
        "z_sem": z_sem.float(),
        "z_sem_global": z_sem_global.float(),
        "material_dist": material_dist,
        "edge_part_idx": edge_part_idx.long(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict physics params from object points + material prior and save interactive checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint")
    parser.add_argument("--object_points", required=True, help="[N,3] or [T,N,3] .npy/.npz")
    parser.add_argument("--object_points_key", default=None)
    parser.add_argument("--train_ready", required=True, help="results/<case>/train/train_ready.pt")
    parser.add_argument("--output", required=True, help="Output .pth for interactive playground")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_knn_topology", action="store_true")
    parser.add_argument("--object_knn", type=int, default=20)
    parser.add_argument("--object_radius", type=float, default=0.02)
    parser.add_argument("--object_max_neighbours", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)

    obj = _load_array(args.object_points, key=args.object_points_key).astype(np.float32)
    if obj.ndim == 3:
        points = obj[0]
    elif obj.ndim == 2:
        points = obj
    else:
        raise ValueError(f"object_points must be [N,3] or [T,N,3], got {obj.shape}")
    if points.shape[-1] != 3:
        raise ValueError(f"object_points last dim must be 3, got {points.shape}")

    train_ready = torch.load(args.train_ready, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    geo_input_dim = int(ckpt["geo_input_dim"])
    dino_dim = int(ckpt["dino_dim"])
    num_materials = int(ckpt.get("num_materials", train_ready["material_distributions"].shape[1]))

    model = create_model_for_type(
        model_type,
        dino_dim=dino_dim,
        geo_input_dim=geo_input_dim,
        num_materials=num_materials,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    topology_cfg = {
        "use_knn_topology": bool(args.use_knn_topology),
        "object_knn": int(args.object_knn),
        "object_radius": float(args.object_radius),
        "object_max_neighbours": int(args.object_max_neighbours),
    }
    feats = build_features(points, train_ready, topology_cfg)

    with torch.no_grad():
        out = forward_model(
            model,
            model_type,
            feats["z_geo"].to(device),
            feats["z_sem"].to(device),
            feats["material_dist"].to(device),
            feats["edge_part_idx"].to(device),
            feats["z_sem_global"].to(device),
            ctrl_sem=torch.zeros((0, dino_dim), device=device),
            ctrl_rest_length=torch.zeros((0, 1), device=device),
            ctrl_part_idx=torch.zeros((0,), dtype=torch.long, device=device),
        )

    pred_logk = out["log_k"].view(-1).cpu()
    spring_Y = pred_logk.exp()

    saved = {
        "spring_Y": spring_Y,
        "collide_elas": out.get("collide_elas", torch.tensor([0.5], device=device)).view(-1).cpu(),
        "collide_fric": out.get("collide_fric", torch.tensor([0.3], device=device)).view(-1).cpu(),
        "collide_object_elas": out.get("collide_object_elas", torch.tensor([0.7], device=device)).view(-1).cpu(),
        "collide_object_fric": out.get("collide_object_fric", torch.tensor([0.3], device=device)).view(-1).cpu(),
        "collision_dist": out.get("collision_dist", torch.tensor([0.02], device=device)).view(-1).cpu(),
        "dashpot_damping": out.get("dashpot_damping", torch.tensor([100.0], device=device)).view(-1).cpu(),
        "drag_damping": out.get("drag_damping", torch.tensor([3.0], device=device)).view(-1).cpu(),
        "num_object_springs": torch.tensor(len(spring_Y), dtype=torch.long),
        "edges": feats["edges"].cpu(),
        "edge_mid": feats["edge_mid"].cpu(),
        "topology_cfg": topology_cfg,
        "source_checkpoint": args.checkpoint,
        "source_train_ready": args.train_ready,
    }
    torch.save(saved, args.output)

    print(f"[done] saved {args.output}")
    print(f"[info] object_points={points.shape[0]} springs={len(spring_Y)}")
    print(
        "[info] spring_Y "
        f"min={spring_Y.min().item():.4f} max={spring_Y.max().item():.4f} mean={spring_Y.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
