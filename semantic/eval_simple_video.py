"""Per-case test eval for SimpleVideoMaterialPhysicsModel checkpoints.

Loads a checkpoint produced by train_model_video_material_simple.py and
re-runs the same test-loader simulation as evaluate(), but logs per-case
track/geo so we can see which scenes the model actually handles well.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from material_param_dataset import MaterialDatasetConfig, create_train_test_dataloaders
from train_models import load_video_frames
from train_model_video_material_simple import (
    SimpleVideoMaterialPhysicsModel,
    forward_case,
    _init_runtime,
    _build_model_logk,
    _apply_model_out_to_sim,
)
from qqtt.utils import cfg, visualize_pc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint .pth path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", choices=("test", "train", "both"), default="test")
    p.add_argument("--out_json", default="")
    p.add_argument("--vis_dir", default="", help="if set, save per-case visualization videos here")
    p.add_argument("--cases", default="", help="comma-separated case names to restrict eval to (optional)")
    return p.parse_args()


def args_from_ckpt(ckpt_args: dict, device: str) -> SimpleNamespace:
    ns = SimpleNamespace(**ckpt_args)
    ns.device = device
    ns.rank = 0
    if getattr(ns, "logk_residual_scale", None) is None:
        ns.logk_residual_scale = 1.0
    if getattr(ns, "logk_soft_clamp", None) is None:
        ns.logk_soft_clamp = 0.25
    return ns


@torch.no_grad()
def eval_loader(model, loader, device, runtimes, args, label, vis_dir="", case_filter=None):
    model.eval()
    per_case = []
    for batch in loader:
        for i, case_name in enumerate(batch["case_name"]):
            if case_filter and case_name not in case_filter:
                continue
            train_frame = int(batch["train_frame"][i].item())
            pixel_values = load_video_frames(
                case_name, args.base_path,
                T=args.num_video_frames,
                image_size=args.videomae_image_size,
                device=device,
            )
            if pixel_values is None:
                print(f"[{label}] {case_name}: NO VIDEO, skip")
                continue
            if case_name not in runtimes:
                runtimes[case_name] = _init_runtime(case_name, train_frame, args)
            runtime = runtimes[case_name]
            sim = runtime.sim

            model_out = forward_case(model, batch, i, device, pixel_values)
            model_logk = _build_model_logk(model_out, runtime, sim, device)
            if model_logk is None:
                print(f"[{label}] {case_name}: model_logk None, skip")
                continue

            _apply_model_out_to_sim(model_out, model_logk, sim)
            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)

            vertices_list = []
            if vis_dir:
                vertices_list.append(wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu())

            sim_total_frames = int(runtime.trainer.dataset.frame_len)
            tr = geo = 0.0
            steps = 0
            for frame_idx in range(1, sim_total_frames if vis_dir else train_frame):
                pure_inf = bool(vis_dir) and frame_idx >= train_frame
                sim.set_controller_target(frame_idx, pure_inference=pure_inf)
                if sim.object_collision_flag:
                    sim.update_collision_graph()
                sim.step()
                sim.calculate_loss()
                if frame_idx < train_frame:
                    tr += wp.to_torch(sim.track_loss, requires_grad=False).item()
                    geo += wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                    steps += 1
                if vis_dir:
                    vertices_list.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu())
                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

            if steps == 0:
                continue
            mt, mg = tr / steps, geo / steps
            tot = float(args.lambda_track) * mt + float(args.lambda_geo) * mg
            per_case.append({
                "case_name": case_name,
                "track": mt,
                "geo": mg,
                "loss": tot,
                "steps": steps,
            })
            print(f"[{label}] {case_name:40s}  track={mt:.6e}  geo={mg:.6e}  loss={tot:.6e}  steps={steps}")

            if vis_dir and vertices_list:
                K_cam = batch["cam0_intrinsics"][i].cpu().numpy()
                w2c = batch["cam0_w2c"][i].cpu().numpy()
                wh = batch["wh"][i]
                W, H = int(wh[0].item()), int(wh[1].item())
                num_all_points = runtime.trainer.num_all_points
                vertices = torch.stack(vertices_list, dim=0)
                Path(vis_dir).mkdir(parents=True, exist_ok=True)
                video_path = os.path.join(vis_dir, f"{label}_{case_name}.mp4")
                try:
                    visualize_pc(
                        vertices[:, :num_all_points, :],
                        runtime.trainer.object_colors,
                        runtime.trainer.controller_points,
                        visualize=False,
                        save_video=True,
                        save_path=video_path,
                        width=W, height=H,
                        intrinsic=K_cam, w2c=w2c,
                        overlay_path=os.path.join(args.base_path, case_name, "color"),
                    )
                    print(f"  [vis] saved {video_path}  ({len(vertices_list)} frames, train_frame={train_frame})")
                except Exception as e:
                    print(f"  [vis] FAILED for {case_name}: {e}")
    return per_case


def main():
    cli = parse_args()
    ckpt = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    epoch = int(ckpt.get("epoch", -1))
    args = args_from_ckpt(ckpt["args"], cli.device)
    print(f"[ckpt] {cli.ckpt} (epoch {epoch})")
    print(f"[cfg] numeric_phys_prior=disabled "
          f"residual_scale={args.logk_residual_scale} soft_clamp={args.logk_soft_clamp}")

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
        results_dir=args.results_dir,
        use_knn_topology=args.use_knn_topology,
        object_knn=args.object_knn,
        object_radius=args.object_radius,
        object_max_neighbours=args.object_max_neighbours,
        controller_radius=args.controller_radius,
        controller_max_neighbours=args.controller_max_neighbours,
    )
    full_dataset, train_loader, test_loader = create_train_test_dataloaders(
        cfg=dataset_cfg,
        batch_size=1,
        num_workers=0,
        train_ratio=args.train_ratio,
        seed=args.seed,
        distributed=False,
    )

    model = SimpleVideoMaterialPhysicsModel(
        videomae_model=args.videomae_model,
        d_motion=args.d_motion,
        d_mat=args.mat_codebook_dim,
        hidden_dim=args.hidden_dim,
        num_materials=args.num_materials,
        logk_residual_scale=args.logk_residual_scale,
        logk_soft_clamp=args.logk_soft_clamp,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    case_filter = set(c.strip() for c in cli.cases.split(",") if c.strip()) if cli.cases else None

    runtimes = {}
    results = {"epoch": epoch, "ckpt": cli.ckpt, "split": cli.split}
    if cli.split in ("test", "both"):
        print("\n=== TEST split ===")
        results["test"] = eval_loader(
            model, test_loader, device, runtimes, args, "test",
            vis_dir=cli.vis_dir, case_filter=case_filter,
        )
    if cli.split in ("train", "both"):
        print("\n=== TRAIN split ===")
        results["train"] = eval_loader(
            model, train_loader, device, runtimes, args, "train",
            vis_dir=cli.vis_dir, case_filter=case_filter,
        )

    for split in ("test", "train"):
        rows = results.get(split)
        if not rows:
            continue
        n = len(rows)
        avg_t = sum(r["track"] for r in rows) / n
        avg_g = sum(r["geo"] for r in rows) / n
        avg_l = sum(r["loss"] for r in rows) / n
        print(f"\n[{split} aggregate over {n} cases] track={avg_t:.6e} geo={avg_g:.6e} loss={avg_l:.6e}")
        results[f"{split}_summary"] = {"n": n, "track": avg_t, "geo": avg_g, "loss": avg_l}

    if cli.out_json:
        Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cli.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {cli.out_json}")


if __name__ == "__main__":
    main()
