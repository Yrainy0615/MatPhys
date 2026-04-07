import argparse
import os
import pickle
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict_physics_params import create_model_for_type, forward_model, build_features
from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset


def load_structure_points(final_data_path: str) -> np.ndarray:
    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    object_points = np.asarray(data["object_points"], dtype=np.float32)
    surface_points = np.asarray(data["surface_points"], dtype=np.float32)
    interior_points = np.asarray(data["interior_points"], dtype=np.float32)

    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(f"object_points must be [T,N,3], got {object_points.shape}")

    return np.concatenate(
        [object_points[0], surface_points, interior_points], axis=0
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Export predicted physics params for an existing case using its structure points"
    )
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_path", default="data/different_types")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--experiments_optimization_dir", default="experiments_optimization")
    parser.add_argument("--sem_cache_dir", default="semantic/cache")
    parser.add_argument("--experiments_dir", default="experiments")
    parser.add_argument(
        "--case_to_material",
        default="semantic/case_to_material_different_types.json",
    )
    parser.add_argument("--train_ready", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    case_dir = os.path.join(args.base_path, args.case_name)
    final_data_path = os.path.join(case_dir, "final_data.pkl")
    if args.train_ready is None:
        args.train_ready = os.path.join(
            args.results_dir, args.case_name, "train", "train_ready.pt"
        )
    if args.output is None:
        args.output = os.path.join(
            args.results_dir, args.case_name, "predicted_physics_from_case.pth"
        )

    opt_path = os.path.join(
        args.experiments_optimization_dir, args.case_name, "optimal_params.pkl"
    )
    topology_cfg = {
        "use_knn_topology": False,
        "object_knn": 20,
        "object_radius": 0.02,
        "object_max_neighbours": 30,
    }
    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt = pickle.load(f)
        topology_cfg["use_knn_topology"] = bool(opt.get("use_knn_topology", False))
        topology_cfg["object_knn"] = int(opt.get("object_knn", topology_cfg["object_knn"]))
        topology_cfg["object_radius"] = float(
            opt.get("object_radius", topology_cfg["object_radius"])
        )
        topology_cfg["object_max_neighbours"] = int(
            opt.get("object_max_neighbours", topology_cfg["object_max_neighbours"])
        )

    points = load_structure_points(final_data_path)

    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
    )
    dataset = MaterialParamDataset(dataset_cfg)
    case_to_idx = {s["case_name"]: i for i, s in enumerate(dataset.samples)}
    if args.case_name not in case_to_idx:
        raise KeyError(f"{args.case_name} not found in MaterialParamDataset")
    sample = dataset[case_to_idx[args.case_name]]

    train_ready = torch.load(args.train_ready, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    geo_input_dim = int(ckpt["geo_input_dim"])
    dino_dim = int(ckpt["dino_dim"])
    num_materials = int(
        ckpt.get("num_materials", train_ready["material_distributions"].shape[1])
    )

    device = torch.device(args.device)
    model = create_model_for_type(
        model_type,
        dino_dim=dino_dim,
        geo_input_dim=geo_input_dim,
        num_materials=num_materials,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

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
            ctrl_sem=sample["ctrl_sem"].to(device),
            ctrl_rest_length=sample["ctrl_rest_length"].to(device),
            ctrl_part_idx=sample["ctrl_part_idx"].to(device),
        )

    pred_logk = out["log_k"].view(-1)
    num_object_springs = int(sample["num_object_springs"].item())
    if pred_logk.numel() != num_object_springs:
        raise ValueError(
            f"pred springs mismatch {pred_logk.numel()} vs {num_object_springs}"
        )
    ctrl_logk = out.get("ctrl_log_k", None)
    if ctrl_logk is not None and ctrl_logk.numel() > 0:
        full_logk = torch.cat([pred_logk.view(-1), ctrl_logk.view(-1)], dim=0)
    else:
        base_spring_y = sample["base_spring_y"].view(-1).to(device)
        base_ctrl_logk = torch.log(base_spring_y[num_object_springs:].clamp_min(1e-8))
        full_logk = torch.cat([pred_logk.view(-1), base_ctrl_logk], dim=0)
    spring_Y = full_logk.exp().cpu()

    saved = {
        "spring_Y": spring_Y,
        "collide_elas": out["collide_elas"].view(-1).cpu(),
        "collide_fric": out["collide_fric"].view(-1).cpu(),
        "collide_object_elas": out["collide_object_elas"].view(-1).cpu(),
        "collide_object_fric": out["collide_object_fric"].view(-1).cpu(),
        "collision_dist": out.get("collision_dist", torch.tensor([0.02])).view(-1).cpu(),
        "dashpot_damping": out.get("dashpot_damping", torch.tensor([100.0])).view(-1).cpu(),
        "drag_damping": out.get("drag_damping", torch.tensor([3.0])).view(-1).cpu(),
        "num_object_springs": torch.tensor(num_object_springs, dtype=torch.long),
        "edges": feats["edges"].cpu(),
        "edge_mid": feats["edge_mid"].cpu(),
        "topology_cfg": topology_cfg,
        "source_checkpoint": args.checkpoint,
        "source_train_ready": args.train_ready,
        "source_final_data": final_data_path,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(saved, args.output)

    print(f"[done] saved {args.output}")
    print(
        f"[info] points={points.shape[0]} springs={len(spring_Y)} "
        f"min={spring_Y.min().item():.4f} max={spring_Y.max().item():.4f} mean={spring_Y.mean().item():.4f}"
    )
    print(f"[info] topology_cfg={topology_cfg}")


if __name__ == "__main__":
    main()
