"""
Run visualization for all cases using a trained edge_level model.

Uses preset collision parameters (no collision optimization). Saves:
  - Sim video per case (like experiments): {output_dir}/{case_name}/train/sim_inference.mp4
  - Optional material visualization when Gaussian PLY exists: {output_dir}/{case_name}/material_visualization.mp4

Usage (from project root):
    python semantic/run_edge_model_vis.py \
        --checkpoint_dir checkpoints/model_comparison/edge_level \
        --output_dir results/model_comparison/edge_level

    # With material visualization (requires gaussian_vis per case):
    python semantic/run_edge_model_vis.py \
        --checkpoint_dir checkpoints/model_comparison/edge_level \
        --output_dir results/model_comparison/edge_level \
        --results_dir results \
        --material_vis
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from typing import Dict

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset
from models import EdgeLevelMaterialPhysics
from train_models import create_model_for_type, forward_model, _load_case_cfg

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg


def main():
    parser = argparse.ArgumentParser(description="Visualize all cases with trained edge_level model")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model_comparison/edge_level",
                        help="Directory containing final_checkpoint.pth")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (overrides checkpoint_dir/final_checkpoint.pth)")
    parser.add_argument("--output_dir", type=str, default="results/model_comparison/edge_level",
                        help="Output root: sim videos and material vis saved under {output_dir}/{case_name}/")
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--sem_cache_dir", type=str, default="semantic/cache")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--experiments_optimization_dir", type=str, default="experiments_optimization")
    parser.add_argument("--case_to_material", type=str, default="semantic/case_to_material_different_types.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--material_vis", action="store_true",
                        help="Run material visualization when Gaussian PLY exists under results_dir")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root for finding gaussian_vis (results_dir/case_name/gaussian_vis/gaussians_vis.ply)")
    parser.add_argument("--use_knn_topology", action="store_true")
    parser.add_argument("--object_knn", type=int, default=20)
    parser.add_argument("--object_radius", type=float, default=0.02)
    parser.add_argument("--object_max_neighbours", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = args.checkpoint or os.path.join(args.checkpoint_dir, "final_checkpoint.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint and build model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "edge_level")
    if model_type != "edge_level":
        print(f"[warn] Checkpoint model_type is '{model_type}'; this script is for edge_level. Proceeding anyway.")
    geo_input_dim = int(ckpt["geo_input_dim"])
    dino_dim = int(ckpt["dino_dim"])
    num_materials = int(ckpt.get("num_materials", 10))
    model = create_model_for_type(
        model_type,
        dino_dim=dino_dim,
        geo_input_dim=geo_input_dim,
        num_materials=num_materials,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    # Dataset for all cases
    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
        use_knn_topology=args.use_knn_topology,
        object_knn=args.object_knn,
        object_radius=args.object_radius,
        object_max_neighbours=args.object_max_neighbours,
    )
    dataset = MaterialParamDataset(dataset_cfg)
    case_names = dataset.case_names
    print(f"[vis] Loaded model from {ckpt_path}")
    print(f"[vis] Cases: {len(case_names)} -> {case_names}")

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, case_name in enumerate(case_names):
        print(f"\n[{idx+1}/{len(case_names)}] {case_name}")
        sample = dataset[idx]
        train_frame = int(sample["train_frame"].item())
        data_path = os.path.join(args.base_path, case_name, "final_data.pkl")
        if not os.path.isfile(data_path):
            print(f"  skip: final_data.pkl not found")
            continue
        base_dir = os.path.join(args.output_dir, case_name)
        os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)

        _load_case_cfg(args.base_path, case_name, args.experiments_optimization_dir)
        trainer = InvPhyTrainerWarp(
            data_path=data_path,
            base_dir=base_dir,
            train_frame=train_frame,
            pure_inference_mode=True,
            device=args.device,
        )
        sim = trainer.simulator
        num_object_springs = trainer.num_object_springs

        # Model forward (preset collision from dataset / optimization)
        z_geo = sample["z_geo"].to(device)
        z_sem = sample["z_sem"].to(device)
        material_dist = sample["material_dist"].to(device)
        edge_part_idx = sample["edge_part_idx"].to(device)
        z_sem_global = sample["z_sem_global"].to(device)
        ctrl_sem = sample["ctrl_sem"].to(device)
        ctrl_rest_length = sample["ctrl_rest_length"].to(device)
        ctrl_part_idx = sample["ctrl_part_idx"].to(device)
        with torch.no_grad():
            out = forward_model(
                model, "edge_level",
                z_geo, z_sem, material_dist, edge_part_idx, z_sem_global,
                ctrl_sem=ctrl_sem, ctrl_rest_length=ctrl_rest_length, ctrl_part_idx=ctrl_part_idx,
            )
        pred_logk = out["log_k"].view(-1)
        if "ctrl_log_k" in out and out["ctrl_log_k"].numel() > 0:
            ctrl_logk = out["ctrl_log_k"].view(-1)
        else:
            base_spring_y = sample["base_spring_y"].to(device)
            ctrl_logk = torch.log(base_spring_y[num_object_springs:].clamp_min(1e-8))
        full_logk = torch.cat([pred_logk, ctrl_logk], dim=0)
        if full_logk.numel() != sim.n_springs:
            print(f"  skip: spring size mismatch {full_logk.numel()} vs {sim.n_springs}")
            continue

        # Set predicted physics parameters
        sim.set_spring_Y(full_logk.detach())
        sim.set_collide(
            out["collide_elas"].detach().view(-1),
            out["collide_fric"].detach().view(-1),
        )
        sim.set_collide_object(
            out["collide_object_elas"].detach().view(-1),
            out["collide_object_fric"].detach().view(-1),
        )

        # Sim video (like experiments)
        video_path = os.path.join(base_dir, "train", "sim_inference.mp4")
        trainer.visualize_sim(save_only=True, video_path=video_path)
        print(f"  saved: {video_path}")

        # Optional: material visualization (Gaussian overlay colored by stiffness)
        if args.material_vis:
            gs_ply = os.path.join(args.results_dir, case_name, "gaussian_vis", "gaussians_vis.ply")
            if os.path.isfile(gs_ply):
                spring_Y = torch.cat([
                    pred_logk.exp().view(-1),
                    base_spring_y[num_object_springs:],
                ])
                fake_ckpt = {
                    "spring_Y": spring_Y.cpu(),
                    "collide_elas": sample["collide_elas"].cpu().view(-1),
                    "collide_fric": sample["collide_fric"].cpu().view(-1),
                    "collide_object_elas": sample["collide_object_elas"].cpu().view(-1),
                    "collide_object_fric": sample["collide_object_fric"].cpu().view(-1),
                    "num_object_springs": num_object_springs,
                }
                with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                    tmp_path = f.name
                try:
                    torch.save(fake_ckpt, tmp_path)
                    trainer.visualize_material(tmp_path, gs_ply, relative_material=True)
                    print(f"  saved: {os.path.join(base_dir, 'material_visualization.mp4')}")
                finally:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
            else:
                print(f"  material_vis: no gaussian PLY at {gs_ply}")

    print(f"\n[done] Outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
