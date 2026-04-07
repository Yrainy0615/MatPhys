"""
Compare predicted physics and topology for single_lift_zebra vs double_lift_zebra
to debug why double_lift sim inference "flies" while single_lift is ok.

Usage (from project root):
    python semantic/compare_zebra_cases.py
    python semantic/compare_zebra_cases.py --checkpoint_dir checkpoints/model_comparison/edge_level
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset
from models import EdgeLevelMaterialPhysics
from train_models import create_model_for_type, forward_model


def _stats(name: str, t: torch.Tensor, extra: str = ""):
    t = t.detach().float()
    if t.numel() == 0:
        return f"  {name}: (empty)"
    s = f"  {name}: shape={tuple(t.shape)} min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f} std={t.std().item():.4f}"
    if extra:
        s += f"  ({extra})"
    return s


def main():
    parser = argparse.ArgumentParser(description="Compare single_lift_zebra vs double_lift_zebra")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model_comparison/edge_level")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--sem_cache_dir", type=str, default="semantic/cache")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--experiments_optimization_dir", type=str, default="experiments_optimization")
    parser.add_argument("--case_to_material", type=str, default="semantic/case_to_material_different_types.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_knn_topology", action="store_true")
    parser.add_argument("--object_knn", type=int, default=20)
    parser.add_argument("--object_radius", type=float, default=0.02)
    parser.add_argument("--object_max_neighbours", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    case_names = ["single_lift_zebra", "double_lift_zebra"]

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

    # Resolve indices
    name_to_idx = {name: idx for idx, name in enumerate(dataset.case_names)}
    for c in case_names:
        if c not in name_to_idx:
            print(f"[error] Case '{c}' not in dataset. Available: {dataset.case_names}")
            sys.exit(1)

    ckpt_path = args.checkpoint or os.path.join(args.checkpoint_dir, "final_checkpoint.pth")
    if not os.path.isfile(ckpt_path):
        print(f"[error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "edge_level")
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

    print("=" * 80)
    print("Comparison: single_lift_zebra vs double_lift_zebra (edge_level model)")
    print("=" * 80)

    results = {}
    for case_name in case_names:
        idx = name_to_idx[case_name]
        sample = dataset[idx]
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
        pred_logk = out["log_k"].squeeze(0)

        E = sample["edges"].shape[0]
        teacher_logk = sample["teacher_logk"].squeeze(0)
        base_spring_y = sample["base_spring_y"]
        k_teacher = teacher_logk.exp()
        k_pred = pred_logk.exp()

        results[case_name] = {
            "n_edges": E,
            "train_frame": int(sample["train_frame"].item()),
            "pred_logk": pred_logk.cpu(),
            "teacher_logk": teacher_logk,
            "k_pred": k_pred.cpu(),
            "k_teacher": k_teacher,
            "z_geo": sample["z_geo"],
            "z_sem": sample["z_sem"],
            "collide_elas": sample["collide_elas"].item(),
            "collide_fric": sample["collide_fric"].item(),
            "collide_object_elas": sample["collide_object_elas"].item(),
            "collide_object_fric": sample["collide_object_fric"].item(),
        }

    # Side-by-side print
    for case_name in case_names:
        r = results[case_name]
        print(f"\n--- {case_name} ---")
        print(f"  n_edges (object springs): {r['n_edges']}")
        print(f"  train_frame (sim steps):  {r['train_frame']}")
        print(_stats("pred_logk", r["pred_logk"]))
        print(_stats("teacher_logk", r["teacher_logk"]))
        print(_stats("k_pred (=exp(pred_logk))", r["k_pred"], "used in sim_inference"))
        print(_stats("k_teacher (from optimization)", r["k_teacher"]))
        print(f"  collide_elas={r['collide_elas']:.4f} collide_fric={r['collide_fric']:.4f}")
        print(f"  collide_object_elas={r['collide_object_elas']:.4f} collide_object_fric={r['collide_object_fric']:.4f}")
        print(_stats("z_geo", r["z_geo"]))
        print(_stats("z_sem (L2 norm per edge)", r["z_sem"].norm(dim=1)))

    # Direct comparison
    print("\n" + "=" * 80)
    print("Direct comparison (possible causes of double_lift 'flying')")
    print("=" * 80)
    s = results["single_lift_zebra"]
    d = results["double_lift_zebra"]

    print(f"\n1. Topology / length:")
    print(f"   single_lift_zebra: {s['n_edges']} edges, train_frame={s['train_frame']}")
    print(f"   double_lift_zebra: {d['n_edges']} edges, train_frame={d['train_frame']}")
    if d["train_frame"] > s["train_frame"]:
        print(f"   -> double_lift runs {d['train_frame'] - s['train_frame']} more sim steps (longer rollout, more room for instability)")

    print(f"\n2. Predicted stiffness k_pred = exp(pred_logk):")
    print(f"   single_lift: k_pred min={s['k_pred'].min().item():.4f} max={s['k_pred'].max().item():.4f} mean={s['k_pred'].mean().item():.4f}")
    print(f"   double_lift: k_pred min={d['k_pred'].min().item():.4f} max={d['k_pred'].max().item():.4f} mean={d['k_pred'].mean().item():.4f}")
    if d["k_pred"].max() > s["k_pred"].max() * 2 or d["k_pred"].min() < s["k_pred"].min() / 2:
        print("   -> double_lift has much larger spread or extreme k values (can cause explosion)")
    if d["k_pred"].min() < 1e-3:
        print("   -> double_lift has very small k (soft springs -> large motion, possible instability)")
    if d["k_pred"].max() > 1e4:
        print("   -> double_lift has very large k (stiff springs -> high frequency, need small dt)")

    print(f"\n3. Teacher (optimization) stiffness k_teacher:")
    print(f"   single_lift: k_teacher mean={s['k_teacher'].mean().item():.4f} std={s['k_teacher'].std().item():.4f}")
    print(f"   double_lift: k_teacher mean={d['k_teacher'].mean().item():.4f} std={d['k_teacher'].std().item():.4f}")
    ratio_pred_teacher_s = (s["k_pred"] / s["k_teacher"].clamp(min=1e-8)).median().item()
    ratio_pred_teacher_d = (d["k_pred"] / d["k_teacher"].clamp(min=1e-8)).median().item()
    print(f"   median(k_pred/k_teacher) single_lift={ratio_pred_teacher_s:.4f} double_lift={ratio_pred_teacher_d:.4f}")
    if ratio_pred_teacher_d > 3 or ratio_pred_teacher_d < 0.33:
        print("   -> double_lift model prediction deviates a lot from teacher (possible domain gap single vs double lift)")

    print(f"\n4. Collision (from optimization checkpoint):")
    print(f"   single_lift: elas={s['collide_elas']:.4f} fric={s['collide_fric']:.4f}")
    print(f"   double_lift: elas={d['collide_elas']:.4f} fric={d['collide_fric']:.4f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
