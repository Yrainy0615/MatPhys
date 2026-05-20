"""Render predicted vs. baseline trajectories with 3D Gaussian Splatting.

Each output frame = alpha * gs_render + (1 - alpha) * original_input_video
so the GT video (background + the actual object being deformed) shows through
semi-transparently — same effect as scripts/make_sidebyside.py's blend panel.

For each requested case:
  1) Load per-case ckpt from per_case_smooth_fitall_<DATE>/<case>/best_checkpoint.pth
  2) Roll out the simulator to get predicted particle positions
  3) Load the case's trained GaussianModel
  4) Use interpolate_motions to deform gaussians per frame (same as gs_render_dynamics)
  5) Render each frame, then cv2.addWeighted with the input video frame
  6) Repeat with experiments/<case>/inference.pkl as the baseline

Outputs land in <out_dir>/<case>/{ours,baseline}.mp4
"""

import argparse
import copy
import glob
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import warp as wp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from material_param_dataset import (
    MaterialDatasetConfig,
    MaterialParamDataset,
    collate_graph_batch,
)
from train_models import load_video_frames
from train_model_video_material_simple import (
    SimpleVideoMaterialPhysicsModel,
    forward_case,
    _init_runtime,
    _build_model_logk,
    _apply_model_out_to_sim,
)
from qqtt.utils import cfg

# 3DGS pieces (same imports trainer_warp / gs_render_dynamics use)
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.dynamic_utils import (
    interpolate_motions,
    knn_weights,
    get_topk_indices,
)
from gs_render import remove_gaussians_with_low_opacity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_root",
        default="checkpoints/per_case_smooth_fitall_20260511",
        help="Per-case ckpt root (<root>/<case>/best_checkpoint.pth). "
             "Ignored if --ckpt_path is set.",
    )
    p.add_argument(
        "--ckpt_path",
        default=None,
        help="Single global ckpt path applied to every requested case. "
             "Overrides --ckpt_root.",
    )
    p.add_argument(
        "--output_suffix",
        default="",
        help="Appended to output filenames, e.g. '_global' -> 'ours_global.mp4'.",
    )
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline render (useful when reusing existing baseline.mp4).",
    )
    p.add_argument(
        "--cases",
        default="double_lift_sloth,single_lift_zebra,single_lift_rope",
    )
    p.add_argument("--baseline_root", default="experiments")
    p.add_argument("--out_dir", default="results/render_gs_blend")
    p.add_argument("--gaussian_root", default="gaussian_output")
    p.add_argument(
        "--gaussian_subdir",
        default="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="GS render opacity in the final composite (input video gets 1-alpha).",
    )
    p.add_argument("--fps", type=int, default=0, help="0 => use cfg.FPS")
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


def build_dataset(args: SimpleNamespace) -> MaterialParamDataset:
    ds_cfg = MaterialDatasetConfig(
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
    return MaterialParamDataset(ds_cfg)


def get_single_batch(dataset: MaterialParamDataset, case_name: str):
    idx = next(
        (i for i, s in enumerate(dataset.samples) if s["case_name"] == case_name),
        None,
    )
    if idx is None:
        raise ValueError(
            f"case_name={case_name!r} not found. "
            f"Available: {[s['case_name'] for s in dataset.samples]}"
        )
    loader = DataLoader(
        Subset(dataset, [idx]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graph_batch,
    )
    return next(iter(loader))


@torch.no_grad()
def rollout_prediction(model, batch, args, device, runtime) -> torch.Tensor:
    """Roll the simulator forward over the full clip; return (T, P, 3) tensor on CPU."""
    case_name = batch["case_name"][0]
    pixel_values = load_video_frames(
        case_name,
        args.base_path,
        T=args.num_video_frames,
        image_size=args.videomae_image_size,
        device=device,
    )
    if pixel_values is None:
        raise RuntimeError(f"No color frames for case {case_name!r}")

    sim = runtime.sim
    model_out = forward_case(model, batch, 0, device, pixel_values)
    model_logk = _build_model_logk(model_out, runtime, sim, device)
    if model_logk is None:
        raise RuntimeError(
            f"Spring count mismatch: model={model_out['log_k'].shape[0]} "
            f"vs sim={runtime.num_object_springs}"
        )

    _apply_model_out_to_sim(model_out, model_logk, sim)
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    sim.set_acc_count(False)

    total_frames = int(runtime.trainer.dataset.frame_len)
    vertices = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()]
    for frame_idx in range(1, total_frames):
        sim.set_controller_target(frame_idx, pure_inference=True)
        if sim.object_collision_flag:
            sim.update_collision_graph()
        sim.step()
        sim.calculate_loss()
        vertices.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu())
        sim.clear_loss()
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
    return torch.stack(vertices, dim=0)


def find_gaussian_ply(case_name: str, gaussian_root: str, gaussian_subdir: str) -> str:
    base = os.path.join(gaussian_root, case_name, gaussian_subdir, "point_cloud")
    if not os.path.isdir(base):
        # fall back to a single match
        glob_root = os.path.join(gaussian_root, case_name, "*", "point_cloud")
        candidates = sorted(glob.glob(glob_root))
        if not candidates:
            raise FileNotFoundError(f"No gaussian point_cloud under {gaussian_root}/{case_name}")
        base = candidates[-1]
    iters = sorted(
        glob.glob(os.path.join(base, "iteration_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not iters:
        raise FileNotFoundError(f"No iteration_* in {base}")
    return os.path.join(iters[-1], "point_cloud.ply")


def make_gs_view(w2c: np.ndarray, intrinsic: np.ndarray, height: int, width: int) -> Camera:
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
    FoVx = focal2fov(K[0, 0], width)
    FoVy = focal2fov(K[1, 1], height)
    return Camera(
        (width, height),
        colmap_id="0000",
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=None,
        invdepthmap=None,
        image_name="0000",
        uid="0000",
        data_device="cuda",
        train_test_exp=None,
        is_test_dataset=None,
        is_test_view=None,
        K=K,
        normal=None,
        depth=None,
        occ_mask=None,
    )


@torch.no_grad()
def deform_gaussians(
    gaussians_base: GaussianModel,
    ctrl_pts: torch.Tensor,
) -> list:
    """Same logic as gs_render_dynamics.rollout(): per-frame deformed gaussian states.

    ctrl_pts: (T, P, 3) tensor on CUDA. Returns a list of (xyz, rot) tuples
    we can swap into a copy of gaussians_base.
    """
    n_steps = ctrl_pts.shape[0]
    xyz_0 = gaussians_base.get_xyz
    rot_0 = gaussians_base.get_rotation

    all_pos = xyz_0.clone()
    all_rot = rot_0.clone()

    init_particle_pos = ctrl_pts[0]
    relations = get_topk_indices(init_particle_pos, K=16)

    frames = [(xyz_0.detach().clone(), rot_0.detach().clone())]
    for i in tqdm(range(1, n_steps), desc="  rollout", dynamic_ncols=True, leave=False):
        prev_particle_pos = ctrl_pts[i - 1]
        cur_particle_pos = ctrl_pts[i]

        chunk_size = 20_000
        num_chunks = (len(all_pos) + chunk_size - 1) // chunk_size
        for j in range(num_chunks):
            s = j * chunk_size
            e = min((j + 1) * chunk_size, len(all_pos))
            pos_c = all_pos[s:e]
            rot_c = all_rot[s:e]
            weights = knn_weights(prev_particle_pos, pos_c, K=16)
            pos_c, rot_c, _ = interpolate_motions(
                bones=prev_particle_pos,
                motions=cur_particle_pos - prev_particle_pos,
                relations=relations,
                weights=weights,
                xyz=pos_c,
                quat=rot_c,
            )
            all_pos[s:e] = pos_c
            all_rot[s:e] = rot_c
        all_rot = torch.nn.functional.normalize(all_rot, dim=-1)
        frames.append((all_pos.detach().clone(), all_rot.detach().clone()))
    return frames


@torch.no_grad()
def render_blend_clip(
    gaussians_base: GaussianModel,
    deformed_frames: list,
    view: Camera,
    background: torch.Tensor,
    case_color_dir: str,
    save_path: str,
    width: int,
    height: int,
    alpha: float,
    fps: int,
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {save_path}")

    for frame_idx, (xyz, rot) in enumerate(
        tqdm(deformed_frames, desc="  render", dynamic_ncols=True, leave=False)
    ):
        gaussians_base._xyz = xyz
        gaussians_base._rotation = rot

        results = render_gaussian(view, gaussians_base, None, background)
        rendering = results["render"]  # (C, H, W); C is 3 or 4 depending on rasterizer
        image = rendering.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()
        gs_rgb = (image[..., :3] * 255).astype(np.uint8)
        gs_bgr = cv2.cvtColor(gs_rgb, cv2.COLOR_RGB2BGR)

        # Load the matching frame from the original input video.
        rgb_path = os.path.join(case_color_dir, f"{frame_idx}.png")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        elif rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height))

        blend = cv2.addWeighted(rgb, 1.0 - alpha, gs_bgr, alpha, 0)
        writer.write(blend)

    writer.release()


def main():
    cli = parse_args()
    cases = [c.strip() for c in cli.cases.split(",") if c.strip()]
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    fps = cli.fps if cli.fps > 0 else int(cfg.FPS)
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")  # white bg for GS

    # Cache the global ckpt+model so we only load once across all cases.
    global_ckpt = None
    global_model = None
    if cli.ckpt_path:
        if not os.path.isfile(cli.ckpt_path):
            raise FileNotFoundError(f"ckpt_path not found: {cli.ckpt_path}")
        global_ckpt = torch.load(cli.ckpt_path, map_location="cpu", weights_only=False)
        print(f"[global ckpt] {cli.ckpt_path} (epoch {global_ckpt.get('epoch')})")

    suffix = cli.output_suffix or ""

    for case_name in cases:
        if cli.ckpt_path:
            ckpt_path = cli.ckpt_path
        else:
            ckpt_path = os.path.join(cli.ckpt_root, case_name, "best_checkpoint.pth")
            if not os.path.isfile(ckpt_path):
                ckpt_path = os.path.join(cli.ckpt_root, case_name, "last_checkpoint.pth")
            if not os.path.isfile(ckpt_path):
                print(f"[skip] no checkpoint for {case_name!r} under {cli.ckpt_root}")
                continue

        print(f"\n=== {case_name} ===")
        print(f"[ckpt] {ckpt_path}")
        ckpt = global_ckpt if global_ckpt is not None else torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        args = args_from_ckpt(ckpt["args"], cli.device)

        dataset = build_dataset(args)
        batch = get_single_batch(dataset, case_name)
        train_frame = int(batch["train_frame"][0].item())
        runtime = _init_runtime(case_name, train_frame, args)

        if global_model is None:
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
            if cli.ckpt_path:
                global_model = model
        else:
            model = global_model

        cfg.chamfer_weight = float(args.lambda_geo)
        cfg.track_weight = float(args.lambda_track)

        # Camera + overlay for view 0
        K_cam = batch["cam0_intrinsics"][0].cpu().numpy()
        w2c = batch["cam0_w2c"][0].cpu().numpy()
        wh = batch["wh"][0]
        W, H = int(wh[0].item()), int(wh[1].item())
        view = make_gs_view(w2c, K_cam, H, W)
        color_dir = os.path.join(args.base_path, case_name, "color", "0")

        # Trained 3DGS for this case
        ply_path = find_gaussian_ply(case_name, cli.gaussian_root, cli.gaussian_subdir)
        print(f"[gs ] {ply_path}")
        gaussians_base = GaussianModel(sh_degree=3)
        gaussians_base.load_ply(ply_path)
        gaussians_base = remove_gaussians_with_low_opacity(gaussians_base)
        # Some checkpoints store isotropic (per-gaussian single scale); enable the
        # property branch that broadcasts (N, 1) → (N, 3) for the rasterizer.
        gaussians_base.isotropic = gaussians_base._scaling.shape[1] == 1
        # Snapshot rest pose so we can reset before each rollout (render_blend_clip
        # mutates ._xyz / ._rotation per frame).
        rest_xyz = gaussians_base.get_xyz.detach().clone()
        rest_rot = gaussians_base.get_rotation.detach().clone()

        num_all_points = runtime.trainer.num_all_points
        case_out = os.path.join(cli.out_dir, case_name)
        Path(case_out).mkdir(parents=True, exist_ok=True)

        # ---- Ours ----
        ours_vertices = rollout_prediction(model, batch, args, device, runtime)
        ours_ctrl = ours_vertices[:, :num_all_points, :].to("cuda")
        gaussians_base._xyz = rest_xyz.clone()
        gaussians_base._rotation = rest_rot.clone()
        deformed = deform_gaussians(gaussians_base, ours_ctrl)
        ours_name = f"ours{suffix}.mp4"
        render_blend_clip(
            gaussians_base=gaussians_base,
            deformed_frames=deformed,
            view=view,
            background=bg,
            case_color_dir=color_dir,
            save_path=os.path.join(case_out, ours_name),
            width=W,
            height=H,
            alpha=cli.alpha,
            fps=fps,
        )
        print(f"  [ours]     -> {os.path.join(case_out, ours_name)}")

        # ---- Baseline ----
        baseline_pkl = os.path.join(cli.baseline_root, case_name, "inference.pkl")
        if cli.skip_baseline:
            print(f"  [baseline] skipped (--skip_baseline)")
        elif os.path.isfile(baseline_pkl):
            with open(baseline_pkl, "rb") as f:
                base_vertices = pickle.load(f)
            base_ctrl = torch.as_tensor(np.asarray(base_vertices), dtype=torch.float32).to("cuda")
            gaussians_base._xyz = rest_xyz.clone()
            gaussians_base._rotation = rest_rot.clone()
            deformed_b = deform_gaussians(gaussians_base, base_ctrl)
            render_blend_clip(
                gaussians_base=gaussians_base,
                deformed_frames=deformed_b,
                view=view,
                background=bg,
                case_color_dir=color_dir,
                save_path=os.path.join(case_out, "baseline.mp4"),
                width=W,
                height=H,
                alpha=cli.alpha,
                fps=fps,
            )
            print(f"  [baseline] -> {os.path.join(case_out, 'baseline.mp4')}")
        else:
            print(f"  [baseline] MISSING {baseline_pkl}, skipping")


if __name__ == "__main__":
    main()
