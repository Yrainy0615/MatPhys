"""
Visualize inference for cases using a trained material prediction model.

Usage:
    python semantic/visualize_inference.py \
        --model_path results/exp_zebra_sloth_dinosor/one-sample/final_model.pth \
        --cases single_lift_zebra double_lift_zebra \
        --save_dir results/exp_zebra_sloth_dinosor/one-sample/vis
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import warp as wp
from tqdm import tqdm

# ---- project imports ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset
from models import EdgeLevelMaterialPhysics, PartLevelMaterialPhysics
from train_models import CaseRuntime, forward_model, _load_case_cfg, create_model_for_type
from qqtt.utils import logger, visualize_pc, cfg


def main():
    parser = argparse.ArgumentParser(description="Visualize inference with predicted material params")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--cases", nargs="+", required=True, help="Case names to visualize")
    parser.add_argument("--base_path", default="data/different_types")
    parser.add_argument("--sem_cache_dir", default="semantic/cache")
    parser.add_argument("--experiments_dir", default="experiments")
    parser.add_argument("--experiments_optimization_dir", default="experiments_optimization")
    parser.add_argument("--case_to_material", default="semantic/case_to_material_different_types.json")
    parser.add_argument("--save_dir", default=None, help="Output directory (default: same dir as model)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.model_path), "vis")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(args.model_path, map_location="cpu")
    model_type = ckpt["model_type"]
    model_args = ckpt.get("args", {})
    print(f"Model type: {model_type}, epoch: {ckpt['epoch']}")

    # Build dataset to get features
    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
    )
    full_dataset = MaterialParamDataset(dataset_cfg)

    # Find case indices
    case_to_idx = {s["case_name"]: i for i, s in enumerate(full_dataset.samples)}
    for c in args.cases:
        if c not in case_to_idx:
            print(f"WARNING: case '{c}' not found in dataset. Available: {list(case_to_idx.keys())}")

    # Get feature dims from first available case
    first_case = args.cases[0]
    sample0 = full_dataset[case_to_idx[first_case]]
    geo_dim = sample0["z_geo"].shape[1]
    dino_dim = sample0["z_sem"].shape[1]
    print(f"Feature dims: geo={geo_dim}, dino={dino_dim}")

    # Create and load model
    num_materials = int(model_args.get("num_materials", ckpt.get("num_materials", 10)))
    model = create_model_for_type(model_type, dino_dim, geo_dim, num_materials=num_materials).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Visualize each case
    for case_name in args.cases:
        print(f"\n{'='*60}")
        print(f"Visualizing: {case_name}")
        print(f"{'='*60}")

        if case_name not in case_to_idx:
            print(f"  Skipping (not in dataset)")
            continue

        sample = full_dataset[case_to_idx[case_name]]
        z_geo = sample["z_geo"].to(device)
        z_sem = sample["z_sem"].to(device)
        material_dist = sample["material_dist"].to(device)
        edge_part_idx = sample["edge_part_idx"].to(device)
        z_sem_global = sample["z_sem_global"].to(device)
        ctrl_sem = sample["ctrl_sem"].to(device)
        ctrl_rest_length = sample["ctrl_rest_length"].to(device)
        ctrl_part_idx = sample["ctrl_part_idx"].to(device)

        # Predict spring_Y
        with torch.no_grad():
            model_out = forward_model(
                model, model_type, z_geo, z_sem, material_dist, edge_part_idx, z_sem_global,
                ctrl_sem=ctrl_sem, ctrl_rest_length=ctrl_rest_length, ctrl_part_idx=ctrl_part_idx,
            )
        pred_logk = model_out["log_k"].view(-1)
        pred_spring_Y = pred_logk.exp()
        print(f"  Predicted spring_Y: min={pred_spring_Y.min().item():.2f}, "
              f"max={pred_spring_Y.max().item():.2f}, "
              f"mean={pred_spring_Y.mean().item():.2f}")

        # Compare with GT
        gt_spring_Y = sample["base_spring_y"]
        nobj = int(sample["num_object_springs"].item())
        gt_obj = gt_spring_Y[:nobj]
        mae = (pred_spring_Y.cpu() - gt_obj).abs().mean().item()
        print(f"  GT spring_Y: min={gt_obj.min().item():.2f}, "
              f"max={gt_obj.max().item():.2f}, mean={gt_obj.mean().item():.2f}")
        print(f"  MAE: {mae:.2f}")

        # Build simulator
        train_frame = int(sample["train_frame"].item())
        rt = CaseRuntime(
            args.base_path, case_name,
            args.experiments_optimization_dir, train_frame, args.device,
        )
        sim = rt.sim

        # Build full spring_Y: predicted object + predicted controller (or GT fallback)
        if "ctrl_log_k" in model_out and model_out["ctrl_log_k"].numel() > 0:
            ctrl_logk = model_out["ctrl_log_k"].view(-1)
        else:
            base_spring_y = sample["base_spring_y"].to(device).view(-1)
            ctrl_logk = torch.log(base_spring_y[nobj:].clamp_min(1e-8))
        full_logk = torch.cat([pred_logk, ctrl_logk])
        sim.set_spring_Y(full_logk.detach())

        # Set predicted collision params
        sim.set_collide(model_out["collide_elas"].detach().view(-1),
                        model_out["collide_fric"].detach().view(-1))
        sim.set_collide_object(model_out["collide_object_elas"].detach().view(-1),
                               model_out["collide_object_fric"].detach().view(-1))
        # Set damping / collision distance from global decoder
        sim.collision_dist = float(model_out["collision_dist"].detach())
        sim.dashpot_damping = float(model_out["dashpot_damping"].detach())
        sim.drag_damping = float(model_out["drag_damping"].detach())

        # Run simulation
        print(f"  Running simulation ({train_frame} frames)...")
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
        vertices = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()]

        for fi in tqdm(range(1, train_frame), desc=f"  Simulating {case_name}"):
            sim.set_controller_target(fi, pure_inference=True)
            if sim.object_collision_flag:
                sim.update_collision_graph()
            sim.step()
            x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
            vertices.append(x.cpu())
            sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

        vertices = torch.stack(vertices, dim=0)
        num_all_points = rt.trainer.num_all_points

        # Save video
        video_path = os.path.join(args.save_dir, f"{case_name}_inference.mp4")
        print(f"  Saving video to {video_path}")
        visualize_pc(
            vertices[:, :num_all_points, :],
            rt.trainer.object_colors,
            rt.trainer.controller_points,
            visualize=False,
            save_video=True,
            save_path=video_path,
        )
        print(f"  Done: {case_name}")

        # Free GPU memory before next case
        del rt, sim
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll visualizations saved to {args.save_dir}")


if __name__ == "__main__":
    main()
