"""
Training script for material physics models with physics simulation.

Uses predicted physics parameters to simulate movement and compares with video input.
Losses: tracking + geometry (chamfer) + render (mask projection)

Model types:
    - edge_level: EdgeLevelMaterialPhysics (per-edge stiffness from material codebook + geometry)
    - part_level: PartLevelMaterialPhysics (per-part stiffness from material codebook only)

Usage:
    python semantic/train_models.py --model_type edge_level --save_dir checkpoints/edge_level
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import warp as wp
from tqdm import tqdm

from material_param_dataset import MaterialDatasetConfig, create_train_test_dataloaders
from models import (
    EdgeLevelMaterialPhysics,
    PartLevelMaterialPhysics,
    print_model_summary,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, visualize_pc


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_case_cfg(base_path: str, case_name: str, experiments_optimization_dir: str) -> None:
    """Load case-specific configuration.

    The simulator must use the same topology as the dataset (old radius-based
    topology from optimal_params.pkl, matching existing checkpoints).
    Disable new KNN/edge-gating/cluster features so the spring count matches.
    """
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    # Force old topology to match dataset / existing checkpoints
    cfg.use_edge_gating = False
    cfg.use_knn_topology = False
    cfg.sem_cache_dir = "__disabled__"

    optimal_path = os.path.join(experiments_optimization_dir, case_name, "optimal_params.pkl")
    if os.path.exists(optimal_path):
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    with open(os.path.join(base_path, case_name, "metadata.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(base_path, case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])
    cfg.WH = data["WH"]
    intrinsics = np.array(data["intrinsics"], dtype=np.float32)
    wh = cfg.WH
    if isinstance(wh, list) and len(wh) == 2:
        width, height = wh
    else:
        width, height = wh[0]
    # Some single-view recon pipelines store intrinsics in normalized [0, 1] image coordinates.
    # Convert them to pixel intrinsics for the simulator / renderer / picking code.
    if np.max(np.abs(intrinsics[:, :2, :])) <= 2.0:
        intrinsics[:, 0, :] *= float(width)
        intrinsics[:, 1, :] *= float(height)
    cfg.intrinsics = intrinsics
    cfg.overlay_path = os.path.join(base_path, case_name, "color")


class CaseRuntime:
    """Runtime environment for a single case with physics simulation."""

    def __init__(
        self,
        base_path: str,
        case_name: str,
        experiments_optimization_dir: str,
        train_frame: int,
        device: str,
    ):
        _load_case_cfg(base_path, case_name, experiments_optimization_dir)
        self.case_name = case_name
        self.case_dir = os.path.join(base_path, case_name)
        self.train_frame = int(train_frame)
        self.device = device
        self.mask_cache: Dict[int, torch.Tensor] = {}

        self.trainer = InvPhyTrainerWarp(
            data_path=os.path.join(base_path, case_name, "final_data.pkl"),
            base_dir=os.path.join("semantic", "runtime", case_name),
            train_frame=self.train_frame,
            pure_inference_mode=True,
            device=device,
        )
        self.sim = self.trainer.simulator
        self.num_object_springs = self.trainer.num_object_springs
        self.num_original_points = self.trainer.num_original_points

    def load_union_mask_cam0(self, frame_idx: int, width: int, height: int) -> torch.Tensor:
        """Load union of all object masks for a frame."""
        if frame_idx in self.mask_cache:
            return self.mask_cache[frame_idx]

        mask_dir = os.path.join(self.case_dir, "mask", "0")
        frame_name = f"{frame_idx}.png"
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*", frame_name)))
        if not mask_paths:
            mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=self.device)
            self.mask_cache[frame_idx] = mask
            return mask

        union_mask = None
        for mask_path in mask_paths:
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
            cur = (img > 0).astype(np.float32)
            union_mask = cur if union_mask is None else np.maximum(union_mask, cur)

        if union_mask is None:
            union_mask = np.zeros((height, width), dtype=np.float32)
        mask = torch.from_numpy(union_mask)[None, None].to(self.device)
        self.mask_cache[frame_idx] = mask
        return mask


def point_mask_render_loss(
    points_world: torch.Tensor,
    mask: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Compute render loss: project points to image and compare with mask."""
    n = points_world.shape[0]
    ones = torch.ones((n, 1), dtype=points_world.dtype, device=points_world.device)
    pts_h = torch.cat([points_world, ones], dim=1)
    cam = pts_h @ w2c.t()
    z = cam[:, 2].clamp(min=1e-6)
    u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]

    u_norm = (u / max(width - 1, 1)) * 2.0 - 1.0
    v_norm = (v / max(height - 1, 1)) * 2.0 - 1.0
    valid = (z > 1e-6) & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
    if valid.sum().item() == 0:
        return torch.zeros((), device=points_world.device, dtype=points_world.dtype)

    grid = torch.stack([u_norm, v_norm], dim=1)[None, :, None, :]
    sampled = F.grid_sample(
        mask,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0, :, 0]
    return (1.0 - sampled[valid]).abs().mean()



def forward_model(
    model: nn.Module,
    model_type: str,
    z_geo: torch.Tensor,
    z_sem: torch.Tensor,
    material_dist: torch.Tensor,
    edge_part_idx: torch.Tensor,
    z_sem_global: torch.Tensor = None,
    # Controller spring features (optional)
    ctrl_sem: torch.Tensor = None,
    ctrl_rest_length: torch.Tensor = None,
    ctrl_part_idx: torch.Tensor = None,
) -> Dict:
    """
    Forward pass through model to get predicted physics parameters.

    Returns:
        dict with:
            - log_k: [E, 1] predicted log stiffness for object springs
            - ctrl_log_k: [C, 1] predicted log stiffness for controller springs (if provided)
            - collide_elas, collide_fric, collide_object_elas, collide_object_fric
    """
    if model_type == "edge_level":
        return model(
            z_geo, z_sem, material_dist, edge_part_idx, z_sem_global,
            ctrl_sem=ctrl_sem, ctrl_rest_length=ctrl_rest_length, ctrl_part_idx=ctrl_part_idx,
        )
    elif model_type == "part_level":
        log_k = model(material_dist, edge_part_idx)
        return {"log_k": log_k}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_global_teacher_loss(batch, idx: int, model_out: Dict, device: torch.device) -> torch.Tensor:
    """Weak supervision for non-differentiable global parameters."""
    teacher_collision_dist = batch["teacher_collision_dist"][idx].to(device).view_as(
        model_out["collision_dist"]
    )
    teacher_drag = batch["teacher_drag_damping"][idx].to(device).view_as(
        model_out["drag_damping"]
    )
    teacher_dashpot = batch["teacher_dashpot_damping"][idx].to(device).view_as(
        model_out["dashpot_damping"]
    )

    loss_collision_dist = F.smooth_l1_loss(
        model_out["collision_dist"] / 0.04,
        teacher_collision_dist / 0.04,
    )
    loss_drag = F.smooth_l1_loss(
        model_out["drag_damping"] / 20.0,
        teacher_drag / 20.0,
    )
    loss_dashpot = F.smooth_l1_loss(
        model_out["dashpot_damping"] / 200.0,
        teacher_dashpot / 200.0,
    )
    return loss_collision_dist + loss_drag + loss_dashpot


def compute_teacher_warmstart_loss(
    pred_logk: torch.Tensor,
    batch,
    idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Weak per-edge teacher regularization for early optimization only."""
    teacher_logk = batch["teacher_logk"][idx].to(device).view_as(pred_logk)
    return F.smooth_l1_loss(pred_logk, teacher_logk)


def compute_warmstart_weight(epoch: int, args) -> float:
    """Linearly decay teacher warm-start weight to zero."""
    base = float(getattr(args, "lambda_teacher_warmstart", 0.0))
    warm_epochs = int(getattr(args, "teacher_warmstart_epochs", 0))
    if base <= 0 or warm_epochs <= 0:
        return 0.0
    if epoch >= warm_epochs:
        return 0.0
    progress = float(epoch) / float(max(warm_epochs, 1))
    return base * max(0.0, 1.0 - progress)


def compute_part_kl_loss(
    model: nn.Module,
    batch,
    idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Part-level semantic consistency: semantic part features -> material distribution."""
    if not hasattr(model, "predict_part_material_logits"):
        return torch.zeros((), device=device)

    part_features = batch["part_features"][idx].to(device)
    target_material_dist = batch["material_dist"][idx].to(device)
    logits = model.predict_part_material_logits(part_features)
    log_probs = F.log_softmax(logits, dim=-1)
    target = target_material_dist.clamp_min(1e-8)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(log_probs, target, reduction="batchmean")


def compute_part_kl_weight(epoch: int, args) -> float:
    base = float(getattr(args, "lambda_part_kl", 0.0))
    start_epoch = int(getattr(args, "part_kl_start_epoch", 0))
    if base <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    return base


def train_epoch_physics(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
    runtimes: Dict[str, CaseRuntime],
    args,
    epoch: int = 0,
) -> Dict[str, float]:
    """Train for one epoch using physics simulation."""
    model.train()

    epoch_loss = 0.0
    epoch_track = 0.0
    epoch_geo = 0.0
    epoch_render = 0.0
    epoch_graphs = 0

    # Count total steps for progress bar
    epoch_total_steps = 0
    for batch in train_loader:
        for i in range(len(batch["case_name"])):
            train_frame_val = int(batch["train_frame"][i].item())
            epoch_total_steps += max(train_frame_val - 1, 1)

    pbar = tqdm(total=max(epoch_total_steps, 1), desc="training", dynamic_ncols=True)

    for batch in train_loader:
        for i in range(len(batch["case_name"])):
            z_geo = batch["z_geo"][i].to(device)
            z_sem = batch["z_sem"][i].to(device)
            z_sem_global = batch["z_sem_global"][i].to(device)
            material_dist = batch["material_dist"][i].to(device)
            edge_part_idx = batch["edge_part_idx"][i].to(device)
            ctrl_sem = batch["ctrl_sem"][i].to(device)
            ctrl_rest_length = batch["ctrl_rest_length"][i].to(device)
            ctrl_part_idx = batch["ctrl_part_idx"][i].to(device)
            case_name = batch["case_name"][i]

            # Get predicted physics params from model
            model_out = forward_model(
                model, model_type, z_geo, z_sem, material_dist, edge_part_idx, z_sem_global,
                ctrl_sem=ctrl_sem, ctrl_rest_length=ctrl_rest_length, ctrl_part_idx=ctrl_part_idx,
            )
            pred_logk = model_out["log_k"]

            # Initialize runtime if needed
            if case_name not in runtimes:
                train_frame_init = int(batch["train_frame"][i].item())
                print(f"[runtime:init] case={case_name} train_frame={train_frame_init}")
                runtimes[case_name] = CaseRuntime(
                    base_path=args.base_path,
                    case_name=case_name,
                    experiments_optimization_dir=args.experiments_optimization_dir,
                    train_frame=train_frame_init,
                    device=args.device,
                )

            runtime = runtimes[case_name]
            sim = runtime.sim

            # Set loss weights
            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)
            cfg.acc_weight = 0.0

            # Build full spring parameters
            num_object_springs = runtime.num_object_springs

            if pred_logk.shape[0] != num_object_springs:
                raise ValueError(
                    f"{case_name}: pred springs mismatch {pred_logk.shape[0]} vs {num_object_springs}"
                )

            # Build full spring logk: object springs + controller springs (all from model)
            ctrl_logk = model_out.get("ctrl_log_k", None)
            if ctrl_logk is not None and ctrl_logk.numel() > 0:
                model_logk = torch.cat([pred_logk.view(-1), ctrl_logk.view(-1)], dim=0)
            else:
                # No controller springs predicted — use baseline for controller part
                base_spring_y = batch["base_spring_y"][i].to(device).view(-1)
                base_ctrl_logk = torch.log(base_spring_y[num_object_springs:].clamp_min(1e-8))
                model_logk = torch.cat([pred_logk.view(-1), base_ctrl_logk], dim=0)

            if model_logk.numel() != sim.n_springs:
                raise ValueError(
                    f"{case_name}: spring size mismatch {model_logk.numel()} vs {sim.n_springs}"
                )

            # Set simulation parameters
            sim.set_spring_Y(model_logk.detach())
            sim.set_collide(
                model_out["collide_elas"].detach().view(-1),
                model_out["collide_fric"].detach().view(-1),
            )
            sim.set_collide_object(
                model_out["collide_object_elas"].detach().view(-1),
                model_out["collide_object_fric"].detach().view(-1),
            )
            # Set damping / collision distance from global decoder
            sim.collision_dist = float(model_out["collision_dist"].detach())
            sim.dashpot_damping = float(model_out["dashpot_damping"].detach())
            sim.drag_damping = float(model_out["drag_damping"].detach())

            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
            sim.set_acc_count(False)

            train_frame = int(batch["train_frame"][i].item())
            K = batch["cam0_intrinsics"][i].to(device)
            w2c = batch["cam0_w2c"][i].to(device)
            wh = batch["wh"][i].to(device)
            width = int(wh[0].item())
            height = int(wh[1].item())

            # Accumulate gradients through simulation
            grad_accum = torch.zeros_like(model_logk)
            grad_collide_elas = torch.zeros_like(model_out["collide_elas"])
            grad_collide_fric = torch.zeros_like(model_out["collide_fric"])
            grad_collide_object_elas = torch.zeros_like(model_out["collide_object_elas"])
            grad_collide_object_fric = torch.zeros_like(model_out["collide_object_fric"])
            graph_geo = 0.0
            graph_track = 0.0
            graph_render = 0.0
            steps = 0
            lambda_render = float(args.lambda_render)

            for frame_idx in range(1, train_frame):
                # Set controller target (external force from hand movement)
                sim.set_controller_target(frame_idx)
                if sim.object_collision_flag:
                    sim.update_collision_graph()

                # Run simulation step and compute loss
                with sim.tape:
                    sim.step()
                    sim.calculate_loss()

                # Get tracking and geometry losses
                track_val = wp.to_torch(sim.track_loss, requires_grad=False).item()
                geo_val = wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                graph_track += track_val
                graph_geo += geo_val

                # Compute render loss with gradient w.r.t. point positions
                pred_points = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=True)
                pred_points_obj = pred_points[: runtime.num_original_points]
                gt_mask = runtime.load_union_mask_cam0(frame_idx, width=width, height=height)
                render_val = point_mask_render_loss(pred_points_obj, gt_mask, K, w2c, width, height)
                graph_render += float(render_val.item())

                # Backward: inject render loss point gradients into warp tape
                if render_val.requires_grad and lambda_render > 0:
                    render_point_grad = torch.autograd.grad(
                        render_val, pred_points, retain_graph=False,
                    )[0] * lambda_render
                    wp_render_grad = wp.from_torch(render_point_grad.contiguous())
                    sim.tape.backward(
                        loss=sim.loss,
                        grads={sim.wp_states[-1].wp_x: wp_render_grad},
                    )
                else:
                    sim.tape.backward(sim.loss)

                # Accumulate gradients for all springs (object + controller)
                grad_full = wp.to_torch(sim.wp_spring_Y.grad, requires_grad=False).detach()
                grad_accum = grad_accum + grad_full
                grad_collide_elas = grad_collide_elas + wp.to_torch(
                    sim.wp_collide_elas.grad, requires_grad=False
                ).detach()
                grad_collide_fric = grad_collide_fric + wp.to_torch(
                    sim.wp_collide_fric.grad, requires_grad=False
                ).detach()
                grad_collide_object_elas = grad_collide_object_elas + wp.to_torch(
                    sim.wp_collide_object_elas.grad, requires_grad=False
                ).detach()
                grad_collide_object_fric = grad_collide_object_fric + wp.to_torch(
                    sim.wp_collide_object_fric.grad, requires_grad=False
                ).detach()

                # Reset for next step
                sim.tape.reset()
                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
                steps += 1
                pbar.update(1)

            if steps == 0:
                pbar.update(1)
                continue

            # Compute mean losses
            mean_track = graph_track / steps
            mean_geo = graph_geo / steps
            mean_render = graph_render / steps

            # Backprop through model using accumulated simulation gradients
            grad_scale = getattr(args, 'grad_scale', 1e4)
            grad_input = (grad_accum / float(steps)) * grad_scale
            grad_collide_elas = grad_collide_elas / float(steps)
            grad_collide_fric = grad_collide_fric / float(steps)
            grad_collide_object_elas = grad_collide_object_elas / float(steps)
            grad_collide_object_fric = grad_collide_object_fric / float(steps)

            optimizer.zero_grad()
            warmstart_weight = compute_warmstart_weight(epoch, args)
            if warmstart_weight > 0:
                warmstart_loss = compute_teacher_warmstart_loss(
                    pred_logk, batch, i, device
                )
                (warmstart_weight * warmstart_loss).backward(retain_graph=True)

            part_kl_weight = compute_part_kl_weight(epoch, args)
            if part_kl_weight > 0:
                part_kl_loss = compute_part_kl_loss(model, batch, i, device)
                (part_kl_weight * part_kl_loss).backward(retain_graph=True)

            lambda_global_teacher = float(getattr(args, "lambda_global_teacher", 0.0))
            if lambda_global_teacher > 0:
                global_teacher_loss = compute_global_teacher_loss(
                    batch, i, model_out, device
                )
                (lambda_global_teacher * global_teacher_loss).backward(retain_graph=True)

            torch.autograd.backward(
                tensors=[
                    model_logk,
                    model_out["collide_elas"],
                    model_out["collide_fric"],
                    model_out["collide_object_elas"],
                    model_out["collide_object_fric"],
                ],
                grad_tensors=[
                    grad_input,
                    grad_collide_elas,
                    grad_collide_fric,
                    grad_collide_object_elas,
                    grad_collide_object_fric,
                ],
            )
            optimizer.step()

            # Accumulate epoch statistics
            total = (
                float(args.lambda_track) * mean_track
                + float(args.lambda_geo) * mean_geo
                + float(args.lambda_render) * mean_render
            )
            epoch_loss += total
            epoch_track += mean_track
            epoch_geo += mean_geo
            epoch_render += mean_render
            epoch_graphs += 1

            pbar.set_postfix(
                total=f"{total:.4f}",
                track=f"{mean_track:.4f}",
                geo=f"{mean_geo:.4f}",
                render=f"{mean_render:.4f}",
            )

    pbar.close()

    n = max(epoch_graphs, 1)
    return {
        "loss": epoch_loss / n,
        "track": epoch_track / n,
        "geo": epoch_geo / n,
        "render": epoch_render / n,
        "num_graphs": epoch_graphs,
    }


@torch.no_grad()
def evaluate_physics(
    model: nn.Module,
    test_loader,
    device: torch.device,
    model_type: str,
    runtimes: Dict[str, CaseRuntime],
    args,
    epoch: int = 0,
    save_dir: str = None,
) -> Dict[str, float]:
    """Evaluate model on test set using physics simulation and save visualization videos."""
    model.eval()

    total_track = 0.0
    total_geo = 0.0
    total_render = 0.0
    total_graphs = 0

    # Create visualization directory
    vis_dir = None
    if save_dir is not None:
        vis_dir = os.path.join(save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

    for batch in test_loader:
        for i in range(len(batch["case_name"])):
            z_geo = batch["z_geo"][i].to(device)
            z_sem = batch["z_sem"][i].to(device)
            z_sem_global = batch["z_sem_global"][i].to(device)
            material_dist = batch["material_dist"][i].to(device)
            edge_part_idx = batch["edge_part_idx"][i].to(device)
            ctrl_sem = batch["ctrl_sem"][i].to(device)
            ctrl_rest_length = batch["ctrl_rest_length"][i].to(device)
            ctrl_part_idx = batch["ctrl_part_idx"][i].to(device)
            case_name = batch["case_name"][i]

            # Get predicted physics params from model
            model_out = forward_model(
                model, model_type, z_geo, z_sem, material_dist, edge_part_idx, z_sem_global,
                ctrl_sem=ctrl_sem, ctrl_rest_length=ctrl_rest_length, ctrl_part_idx=ctrl_part_idx,
            )
            pred_logk = model_out["log_k"]

            # Initialize runtime if needed
            if case_name not in runtimes:
                train_frame_init = int(batch["train_frame"][i].item())
                print(f"[runtime:init:eval] case={case_name} train_frame={train_frame_init}")
                runtimes[case_name] = CaseRuntime(
                    base_path=args.base_path,
                    case_name=case_name,
                    experiments_optimization_dir=args.experiments_optimization_dir,
                    train_frame=train_frame_init,
                    device=args.device,
                )

            runtime = runtimes[case_name]
            sim = runtime.sim

            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)
            cfg.acc_weight = 0.0

            num_object_springs = runtime.num_object_springs

            if pred_logk.shape[0] != num_object_springs:
                continue

            # Controller springs: predicted if available, else fallback to baseline
            if "ctrl_log_k" in model_out and model_out["ctrl_log_k"].numel() > 0:
                ctrl_logk = model_out["ctrl_log_k"].view(-1)
            else:
                base_spring_y = batch["base_spring_y"][i].to(device).view(-1)
                ctrl_logk = torch.log(base_spring_y[num_object_springs:].clamp_min(1e-8))

            full_logk = torch.cat([pred_logk.view(-1), ctrl_logk], dim=0)
            if full_logk.numel() != sim.n_springs:
                continue

            sim.set_spring_Y(full_logk.detach())
            sim.set_collide(
                model_out["collide_elas"].detach().view(-1),
                model_out["collide_fric"].detach().view(-1),
            )
            sim.set_collide_object(
                model_out["collide_object_elas"].detach().view(-1),
                model_out["collide_object_fric"].detach().view(-1),
            )
            # Set damping / collision distance from global decoder
            sim.collision_dist = float(model_out["collision_dist"].detach())
            sim.dashpot_damping = float(model_out["dashpot_damping"].detach())
            sim.drag_damping = float(model_out["drag_damping"].detach())

            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
            sim.set_acc_count(False)

            train_frame = int(batch["train_frame"][i].item())
            K = batch["cam0_intrinsics"][i].to(device)
            w2c = batch["cam0_w2c"][i].to(device)
            wh = batch["wh"][i].to(device)
            width = int(wh[0].item())
            height = int(wh[1].item())

            graph_geo = 0.0
            graph_track = 0.0
            graph_render = 0.0
            steps = 0

            # Collect vertices for visualization
            vertices_list = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()]

            for frame_idx in range(1, train_frame):
                sim.set_controller_target(frame_idx)
                if sim.object_collision_flag:
                    sim.update_collision_graph()

                # Run simulation (no gradient needed)
                sim.step()
                sim.calculate_loss()

                track_val = wp.to_torch(sim.track_loss, requires_grad=False).item()
                geo_val = wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                graph_track += track_val
                graph_geo += geo_val

                pred_points = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
                pred_points = pred_points[: runtime.num_original_points]
                gt_mask = runtime.load_union_mask_cam0(frame_idx, width=width, height=height)
                render_val = point_mask_render_loss(pred_points, gt_mask, K, w2c, width, height)
                graph_render += float(render_val.item())

                # Save vertices for visualization
                vertices_list.append(
                    wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu()
                )

                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
                steps += 1

            if steps > 0:
                total_track += graph_track / steps
                total_geo += graph_geo / steps
                total_render += graph_render / steps
                total_graphs += 1

                # Save visualization video
                if vis_dir is not None:
                    vertices = torch.stack(vertices_list, dim=0)
                    num_all_points = runtime.trainer.num_all_points
                    video_path = os.path.join(
                        vis_dir, f"sim_inference_{case_name}_epoch{epoch:04d}.mp4"
                    )
                    try:
                        visualize_pc(
                            vertices[:, :num_all_points, :],
                            runtime.trainer.object_colors,
                            runtime.trainer.controller_points,
                            visualize=False,
                            save_video=True,
                            save_path=video_path,
                            width=width,
                            height=height,
                            intrinsic=K.detach().cpu().numpy(),
                            w2c=w2c.detach().cpu().numpy(),
                            overlay_path=os.path.join(args.base_path, case_name, "color"),
                        )
                        print(f"  [vis] Saved {video_path}")
                    except Exception as e:
                        print(f"  [vis] Failed to save {video_path}: {e}")

    n = max(total_graphs, 1)
    return {
        "track": total_track / n,
        "geo": total_geo / n,
        "render": total_render / n,
        "total": (
            args.lambda_track * total_track / n
            + args.lambda_geo * total_geo / n
            + args.lambda_render * total_render / n
        ),
        "num_graphs": total_graphs,
    }


def create_model_for_type(
    model_type: str,
    dino_dim: int = 1024,
    geo_input_dim: int = 10,
    num_materials: int = 10,
) -> nn.Module:
    """Create model based on type."""
    if model_type == "edge_level":
        return EdgeLevelMaterialPhysics(
            sem_dim=dino_dim,
            dino_dim=dino_dim,
            geo_input_dim=geo_input_dim,
            num_materials=num_materials,
        )
    elif model_type == "part_level":
        return PartLevelMaterialPhysics(num_materials=num_materials)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--sem_cache_dir", type=str, default="semantic/cache")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument(
        "--experiments_optimization_dir",
        type=str,
        default="experiments_optimization",
    )
    parser.add_argument("--case_to_material", type=str, default="semantic/case_to_material_different_types.json")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory with extract_material_parts.py outputs (train_ready.pt)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_type",
        type=str,
        default="edge_level",
        choices=["edge_level", "part_level"],
    )
    parser.add_argument("--num_materials", type=int, default=10,
                        help="Number of material codes in codebook")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of cases for training (default 0.8 = 8:2 train/test)")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate on test set every N epochs")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_knn_topology", action="store_true")
    parser.add_argument("--object_knn", type=int, default=30)
    parser.add_argument("--object_radius", type=float, default=0.02)
    parser.add_argument("--object_max_neighbours", type=int, default=30)

    # Loss weights
    parser.add_argument("--lambda_render", type=float, default=1.0)
    parser.add_argument("--lambda_track", type=float, default=1.0)
    parser.add_argument("--lambda_geo", type=float, default=1.0)

    # Single case training (for testing model capacity)
    parser.add_argument("--case_name", type=str, default=None,
                        help="Train on a single case only (for testing model capacity)")

    # Gradient scaling (warp simulation gradients are very small)
    parser.add_argument("--grad_scale", type=float, default=1e3,
                        help="Scale factor for simulation gradients (default: 1e4)")
    parser.add_argument(
        "--lambda_global_teacher",
        type=float,
        default=0.05,
        help="Weak teacher regularization for non-differentiable global params",
    )
    parser.add_argument(
        "--lambda_teacher_warmstart",
        type=float,
        default=0.0,
        help="Initial weight of per-edge teacher warm-start loss",
    )
    parser.add_argument(
        "--teacher_warmstart_epochs",
        type=int,
        default=0,
        help="Number of early epochs to linearly decay teacher warm-start loss",
    )
    parser.add_argument(
        "--lambda_part_kl",
        type=float,
        default=0.0,
        help="Weight for part-level semantic-to-material KL consistency",
    )
    parser.add_argument(
        "--part_kl_start_epoch",
        type=int,
        default=0,
        help="Epoch to start part-level KL consistency",
    )

    args = parser.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join("semantic", "runtime"), exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Training Model: {args.model_type}")
    print(f"Loss weights: track={args.lambda_track}, geo={args.lambda_geo}, render={args.lambda_render}")
    if args.case_name:
        print(f"Single case mode: {args.case_name}")
    print(f"{'='*60}\n")

    # Create dataset with train/test split
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
    )

    # Single case mode for testing model capacity
    if args.case_name:
        from material_param_dataset import MaterialParamDataset, collate_graph_batch
        full_dataset = MaterialParamDataset(dataset_cfg)

        # Filter to single case
        case_idx = None
        for idx, sample in enumerate(full_dataset.samples):
            if sample["case_name"] == args.case_name:
                case_idx = idx
                break

        if case_idx is None:
            available = [s["case_name"] for s in full_dataset.samples]
            raise ValueError(f"Case '{args.case_name}' not found. Available: {available}")

        # Create single-case dataset
        single_case_samples = [full_dataset.samples[case_idx]]
        full_dataset.samples = single_case_samples

        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graph_batch,
        )
        test_loader = train_loader  # Same for single case
        print(f"[Single Case] Training on: {args.case_name}")
    else:
        # Random 8:2 split of cases (train_ratio=0.8, overridable via --train_ratio)
        full_dataset, train_loader, test_loader = create_train_test_dataloaders(
            cfg=dataset_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )

    # Get dimensions from dataset
    sample = full_dataset[0]
    geo_input_dim = sample["z_geo"].shape[1]
    dino_dim = sample["z_sem"].shape[1]

    print(f"Geometry feature dim: {geo_input_dim}")
    print(f"DINO feature dim: {dino_dim}")

    num_materials = sample["material_dist"].shape[1]
    print(f"Number of material codes: {num_materials}")

    # Create model
    model = create_model_for_type(
        args.model_type,
        dino_dim=dino_dim,
        geo_input_dim=geo_input_dim,
        num_materials=num_materials,
    )
    model = model.to(device)
    print_model_summary(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Shared runtimes for train and test
    runtimes: Dict[str, CaseRuntime] = {}

    best_test_loss = float('inf')
    best_epoch = 0
    train_history = []
    test_history = []

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch_physics(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            model_type=args.model_type,
            runtimes=runtimes,
            args=args,
            epoch=epoch,
        )
        train_history.append(train_metrics)

        print(
            f"[Epoch {epoch+1:04d}] Train: "
            f"loss={train_metrics['loss']:.6f} "
            f"track={train_metrics['track']:.6f} "
            f"geo={train_metrics['geo']:.6f} "
            f"render={train_metrics['render']:.6f}"
        )

        # Evaluate on test set periodically
        if epoch % args.eval_every == 0 or (epoch + 1) == args.epochs:
            test_metrics = evaluate_physics(
                model=model,
                test_loader=test_loader,
                device=device,
                model_type=args.model_type,
                runtimes=runtimes,
                args=args,
                epoch=epoch + 1,
                save_dir=args.save_dir,
            )
            test_history.append({"epoch": epoch + 1, **test_metrics})

            print(
                f"[Epoch {epoch+1:04d}] Test:  "
                f"total={test_metrics['total']:.6f} "
                f"track={test_metrics['track']:.6f} "
                f"geo={test_metrics['geo']:.6f} "
                f"render={test_metrics['render']:.6f}"
            )

            if test_metrics["total"] < best_test_loss:
                best_test_loss = test_metrics["total"]
                best_epoch = epoch + 1

        # Save checkpoint every 50 epochs
        if (epoch + 1) % args.eval_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_type": args.model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "geo_input_dim": geo_input_dim,
                "dino_dim": dino_dim,
                "num_materials": num_materials,
                "train_metrics": train_metrics,
                "best_test_loss": best_test_loss,
                "best_epoch": best_epoch,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")

    final_metrics = evaluate_physics(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type=args.model_type,
        runtimes=runtimes,
        args=args,
        epoch=args.epochs,
        save_dir=args.save_dir,
    )

    print(f"Model Type: {args.model_type}")
    print(f"Test Total Loss: {final_metrics['total']:.6f}")
    print(f"  - Track Loss:  {final_metrics['track']:.6f}")
    print(f"  - Geo Loss:    {final_metrics['geo']:.6f}")
    print(f"  - Render Loss: {final_metrics['render']:.6f}")
    print(f"Best Total Loss: {best_test_loss:.6f} (Epoch {best_epoch})")

    # Save final checkpoint only
    ckpt = {
        "epoch": args.epochs,
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "geo_input_dim": geo_input_dim,
        "dino_dim": dino_dim,
        "num_materials": num_materials,
        "final_metrics": final_metrics,
        "best_test_loss": best_test_loss,
        "best_epoch": best_epoch,
        "args": vars(args),
    }

    ckpt_path = os.path.join(args.save_dir, "final_checkpoint.pth")
    torch.save(ckpt, ckpt_path)
    print(f"\nSaved final checkpoint to: {ckpt_path}")

    # Save metrics to JSON for comparison
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "epochs": args.epochs,
            "final_metrics": final_metrics,
            "best_test_loss": best_test_loss,
            "best_epoch": best_epoch,
            "train_history": train_history,
            "test_history": test_history,
        }, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
