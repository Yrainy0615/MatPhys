"""Run a pretrained SimpleVideoMaterialPhysicsModel on one case and dump
its predicted physics parameters in the format expected by
interactive_playground.py.

Outputs written:
    experiments_optimization/<case>/optimal_params.pkl     # first-stage globals
    experiments/<case>/train/best_pretrained.pth           # per-edge spring_Y + collisions

Usage:
    python semantic/dump_physics_params.py \\
        --ckpt checkpoints/.../checkpoint_epoch_0200.pth \\
        --case_name monkey \\
        --base_path data/single_view \\
        --case_to_material semantic/case_to_material_monkey_only.json
"""
import argparse
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch

SEMANTIC_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SEMANTIC_DIR, ".."))
for p in (SEMANTIC_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from material_param_dataset import MaterialDatasetConfig, create_train_test_dataloaders
from train_model_video_material_simple import (
    SimpleVideoMaterialPhysicsModel,
    _init_runtime,
    forward_case,
    load_video_frames,
)


def args_from_ckpt(ckpt_args: dict, overrides: dict) -> SimpleNamespace:
    ns = SimpleNamespace(**ckpt_args)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--case_name", required=True)
    p.add_argument("--base_path", required=True,
                   help="Data root (e.g. data/single_view)")
    p.add_argument("--case_to_material", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--sem_cache_dir", default="semantic/cache")
    p.add_argument("--experiments_dir", default="experiments")
    p.add_argument("--experiments_optimization_dir", default="experiments_optimization")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_tag", default="pretrained",
                   help="best_<tag>.pth (under experiments/<case>/train/)")
    p.add_argument("--topology", choices=("ckpt", "knn", "radius"), default="ckpt",
                   help="Spring graph topology. 'ckpt' (default) uses the topology the model was "
                        "trained with — that's the only one whose per-edge predictions are in-distribution. "
                        "'knn'/'radius' force-override and are only useful for diagnostics.")
    p.add_argument("--logk_base", type=float, default=None,
                   help="Override logk_base (default: model constructor default).")
    p.add_argument("--logk_min", type=float, default=None)
    p.add_argument("--logk_max", type=float, default=None)
    cli = p.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    print(f"[ckpt] {cli.ckpt} epoch={ckpt.get('epoch')}")

    overrides = {
        "base_path": cli.base_path,
        "experiments_dir": cli.experiments_dir,
        "experiments_optimization_dir": cli.experiments_optimization_dir,
        "case_to_material": cli.case_to_material,
        "results_dir": cli.results_dir,
        "sem_cache_dir": cli.sem_cache_dir,
        "device": cli.device,
    }
    if cli.topology == "knn":
        overrides["use_knn_topology"] = True
    elif cli.topology == "radius":
        overrides["use_knn_topology"] = False
    args = args_from_ckpt(ckpt_args, overrides)

    # Build dataset (single-case selection happens via case_to_material file).
    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
        results_dir=args.results_dir,
        use_knn_topology=getattr(args, "use_knn_topology", False),
        object_knn=getattr(args, "object_knn", 20),
        object_radius=getattr(args, "object_radius", 0.02),
        object_max_neighbours=getattr(args, "object_max_neighbours", 30),
        controller_radius=getattr(args, "controller_radius", 0.04),
        controller_max_neighbours=getattr(args, "controller_max_neighbours", 50),
    )
    full_dataset, train_loader, _test_loader = create_train_test_dataloaders(
        cfg=dataset_cfg,
        batch_size=1,
        num_workers=0,
        train_ratio=1.0,
        seed=getattr(args, "seed", 42),
        distributed=False,
    )

    # Find the sample for our case
    target_batch = None
    target_idx = None
    for batch in train_loader:
        for i, c in enumerate(batch["case_name"]):
            if c == cli.case_name:
                target_batch, target_idx = batch, i
                break
        if target_batch is not None:
            break
    if target_batch is None:
        raise RuntimeError(
            f"case '{cli.case_name}' not found in dataset built from {cli.case_to_material}"
        )

    # Build model and load weights
    model_kwargs = dict(
        videomae_model=args.videomae_model,
        d_motion=args.d_motion,
        d_mat=args.mat_codebook_dim,
        hidden_dim=args.hidden_dim,
        num_materials=args.num_materials,
        logk_residual_scale=args.logk_residual_scale,
        logk_soft_clamp=args.logk_soft_clamp,
    )
    if cli.logk_base is not None:
        model_kwargs["logk_base"] = cli.logk_base
    if cli.logk_min is not None:
        model_kwargs["logk_min"] = cli.logk_min
    if cli.logk_max is not None:
        model_kwargs["logk_max"] = cli.logk_max
    model = SimpleVideoMaterialPhysicsModel(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Build runtime to discover spring topology (object + controller springs)
    train_frame = int(target_batch["train_frame"][target_idx].item())
    runtime = _init_runtime(cli.case_name, train_frame, args)
    sim = runtime.sim
    n_obj_springs = runtime.num_object_springs
    n_ctrl_springs = sim.n_springs - n_obj_springs
    print(f"[runtime] springs: object={n_obj_springs}, controller={n_ctrl_springs}, total={sim.n_springs}")

    # Forward
    pixel_values = load_video_frames(
        cli.case_name, args.base_path,
        T=args.num_video_frames, image_size=args.videomae_image_size, device=device,
    )
    if pixel_values is None:
        raise RuntimeError(f"could not load video frames for {cli.case_name}")
    with torch.no_grad():
        out = forward_case(model, target_batch, target_idx, device, pixel_values)

    log_k_obj = out["log_k"].detach().cpu().numpy().astype(np.float64)
    spring_Y_obj = np.exp(log_k_obj).astype(np.float32)
    print(f"[forward] log_k(obj): mean={log_k_obj.mean():.3f}  spring_Y mean={spring_Y_obj.mean():.1f}")

    if "ctrl_log_k" in out:
        ctrl_log_k = out["ctrl_log_k"].detach().cpu().numpy().astype(np.float64)
        spring_Y_ctrl = np.exp(ctrl_log_k).astype(np.float32)
    else:
        spring_Y_ctrl = np.zeros(n_ctrl_springs, dtype=np.float32)
        print("[forward] no ctrl_log_k in output; controller springs set to 0")

    # Sanity: spring counts
    if spring_Y_obj.shape[0] != n_obj_springs:
        raise RuntimeError(
            f"object spring count mismatch: model {spring_Y_obj.shape[0]} vs sim {n_obj_springs}"
        )
    if spring_Y_ctrl.shape[0] != n_ctrl_springs:
        print(f"[warn] controller spring count mismatch: model {spring_Y_ctrl.shape[0]} vs sim {n_ctrl_springs}; padding/clipping")
        if spring_Y_ctrl.shape[0] > n_ctrl_springs:
            spring_Y_ctrl = spring_Y_ctrl[:n_ctrl_springs]
        else:
            spring_Y_ctrl = np.concatenate([
                spring_Y_ctrl,
                np.full(n_ctrl_springs - spring_Y_ctrl.shape[0],
                        float(spring_Y_obj.mean()), dtype=np.float32),
            ])

    spring_Y_full = np.concatenate([spring_Y_obj, spring_Y_ctrl], axis=0)

    # Pull scalar global params (move tensors to cpu floats)
    def _scalar(name):
        return float(out[name].detach().cpu().item())

    collide_elas = _scalar("collide_elas")
    collide_fric = _scalar("collide_fric")
    collide_obj_elas = _scalar("collide_object_elas")
    collide_obj_fric = _scalar("collide_object_fric")
    collision_dist = _scalar("collision_dist")
    drag_damping = _scalar("drag_damping")
    dashpot_damping = _scalar("dashpot_damping")
    print(
        f"[forward] collide_elas={collide_elas:.3f} fric={collide_fric:.3f} "
        f"obj_elas={collide_obj_elas:.3f} obj_fric={collide_obj_fric:.3f} "
        f"col_dist={collision_dist:.4f} drag={drag_damping:.2f} dashpot={dashpot_damping:.2f}"
    )

    # ---- Write playground-format optimal_params.pkl ----
    opt_dir = os.path.join(cli.experiments_optimization_dir, cli.case_name)
    os.makedirs(opt_dir, exist_ok=True)
    optimal_params = {
        "global_spring_Y": np.float64(spring_Y_obj.mean()),
        "object_radius": np.float64(getattr(args, "object_radius", 0.02)),
        "object_max_neighbours": int(getattr(args, "object_max_neighbours", 30)),
        "controller_radius": np.float64(getattr(args, "controller_radius", 0.04)),
        "controller_max_neighbours": int(getattr(args, "controller_max_neighbours", 50)),
        # Self-describe topology so playground/inference rebuild the exact spring
        # graph that was used to dump these params. Without this, downstream
        # scripts fall back to whatever configs/*.yaml currently says and
        # silently mismatch n_springs.
        "use_knn_topology": bool(getattr(args, "use_knn_topology", False)),
        "object_knn": int(getattr(args, "object_knn", 30)),
        "controller_knn": int(getattr(args, "controller_knn", 30)),
        "collide_elas": np.float32(collide_elas),
        "collide_fric": np.float64(collide_fric),
        "collide_object_elas": np.float32(collide_obj_elas),
        "collide_object_fric": np.float64(collide_obj_fric),
        "collision_dist": np.float64(collision_dist),
        "drag_damping": np.float64(drag_damping),
        "dashpot_damping": np.float64(dashpot_damping),
    }
    opt_path = os.path.join(opt_dir, "optimal_params.pkl")
    with open(opt_path, "wb") as f:
        pickle.dump(optimal_params, f)
    print(f"[write] {opt_path}")

    # ---- Write playground-format best_<tag>.pth ----
    train_dir = os.path.join(cli.experiments_dir, cli.case_name, "train")
    os.makedirs(train_dir, exist_ok=True)
    best_pkl = {
        "epoch": int(ckpt.get("epoch", -1)),
        "num_object_springs": int(n_obj_springs),
        "spring_Y": torch.tensor(spring_Y_full, dtype=torch.float32),
        "collide_elas": torch.tensor([collide_elas], dtype=torch.float32),
        "collide_fric": torch.tensor([collide_fric], dtype=torch.float32),
        "collide_object_elas": torch.tensor([collide_obj_elas], dtype=torch.float32),
        "collide_object_fric": torch.tensor([collide_obj_fric], dtype=torch.float32),
        "optimizer_state_dict": {},
    }
    best_path = os.path.join(train_dir, f"best_{cli.out_tag}.pth")
    torch.save(best_pkl, best_path)
    print(f"[write] {best_path}")
    print("DONE")


if __name__ == "__main__":
    main()
