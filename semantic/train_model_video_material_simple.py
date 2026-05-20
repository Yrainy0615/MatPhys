"""
Train a simplified video/material physics model.

Design:
  - Frozen VideoMAE backbone extracts motion evidence.
  - A small trainable projector maps pooled VideoMAE tokens to z_motion.
  - DINO features are not consumed by the physics decoder; they are assumed to
    have already been used upstream for part segmentation and material queries.
  - A learnable material codebook gives part-level material embeddings.
  - Edge stiffness is predicted from geometry + part material, with a global
    motion/material modulation.
  - Global collision/friction/damping uses projected video feature + pooled
    material embedding + scene geometry stats.

Default training uses Warp rollout tracking/chamfer losses as the main
supervision. VLM material class probabilities are used only as input to a
learnable material codebook; VLM/GPT numeric physics priors are not used as
log-k targets. Direct-optimization teacher targets are optional and disabled
by default.
"""

import argparse
import datetime
import math
import os
import sys
import warnings
from typing import Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import warp as wp
from tqdm import tqdm

from material_param_dataset import MaterialDatasetConfig, create_train_test_dataloaders

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_models import (
    CaseRuntime,
    _find_latest_checkpoint,
    _is_distributed,
    _all_reduce_mean,
    _unwrap_model,
    _clamp_log_stiffness,
    _is_finite_tensor,
    _make_gs_view,
    gaussian_render_l1_loss,
    load_video_frames,
    set_all_seeds,
    compute_part_physics_prior_loss,
    compute_global_physics_prior_loss,
)
from qqtt.utils import cfg, visualize_pc


LOG_K_MIN = float(np.log(1e3))
LOG_K_MAX = float(np.log(1e5))
LOG_K_SPAN = LOG_K_MAX - LOG_K_MIN


_FRAME_LEN_CACHE: Dict[str, int] = {}


def _resolve_train_frame(args, case_name: str, batch_value: int) -> int:
    """Return the effective frame count for simulation rollout.

    When --fit_all_frames is set, read frame_len from split.json so the
    trainer fits the test frames in addition to the train split.
    """
    if not getattr(args, "fit_all_frames", False):
        return batch_value
    cached = _FRAME_LEN_CACHE.get(case_name)
    if cached is not None:
        return cached
    import json as _json
    split_path = os.path.join(args.base_path, case_name, "split.json")
    with open(split_path, "r") as f:
        split = _json.load(f)
    frame_len = int(split.get("frame_len", batch_value))
    _FRAME_LEN_CACHE[case_name] = frame_len
    return frame_len


def _sigmoid_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.sigmoid(x) * (hi - lo) + lo


def _soft_clamp(x: torch.Tensor, lo: float, hi: float, softness: float) -> torch.Tensor:
    if softness <= 0:
        return x.clamp(lo, hi)
    return lo + softness * F.softplus((x - lo) / softness) - softness * F.softplus((x - hi) / softness)


class FrozenVideoMAEMotionEncoder(nn.Module):
    """Frozen VideoMAE backbone plus a small trainable motion projector."""

    def __init__(self, model_name: str, d_motion: int = 128, projector_hidden: int = 256):
        super().__init__()
        from transformers import VideoMAEModel

        self.backbone = VideoMAEModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        hidden_size = self.backbone.config.hidden_size
        self.tubelet_size = getattr(self.backbone.config, "tubelet_size", 2)
        self.patch_size = getattr(self.backbone.config, "patch_size", 16)

        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projector_hidden),
            nn.SiLU(),
            nn.Linear(projector_hidden, d_motion),
            nn.LayerNorm(d_motion),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = pixel_values.shape
        n_patches = (t // self.tubelet_size) * (h // self.patch_size) * (w // self.patch_size)
        mask = torch.zeros(b, n_patches, dtype=torch.bool, device=pixel_values.device)
        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values, bool_masked_pos=mask)
            pooled = out.last_hidden_state.mean(dim=1)
        return self.projector(pooled).squeeze(0)


class MaterialCodebook(nn.Module):
    def __init__(self, num_materials: int = 10, d_mat: int = 32):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_materials, d_mat) * 0.02)

    def forward(self, material_dist: torch.Tensor) -> torch.Tensor:
        return material_dist @ self.codebook


class SimpleVideoMaterialPhysicsModel(nn.Module):
    def __init__(
        self,
        videomae_model: str = "MCG-NJU/videomae-base",
        d_motion: int = 128,
        d_mat: int = 32,
        hidden_dim: int = 256,
        geo_input_dim: int = 11,
        geo_dim: int = 32,
        geo_stats_dim: int = 6,
        num_materials: int = 10,
        logk_base: float = 3e4,
        logk_min: float = 1e3,
        logk_max: float = 1e5,
        logk_residual_scale: float = 1.0,
        logk_soft_clamp: float = 0.25,
    ):
        super().__init__()
        self.logk_base = float(logk_base)
        self.logk_min = float(logk_min)
        self.logk_max = float(logk_max)
        self.logk_residual_scale = float(logk_residual_scale)
        self.logk_soft_clamp = float(logk_soft_clamp)
        self.motion_encoder = FrozenVideoMAEMotionEncoder(
            videomae_model, d_motion=d_motion, projector_hidden=hidden_dim,
        )
        self.material_codebook = MaterialCodebook(num_materials, d_mat)
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, geo_dim),
        )
        self.rest_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, geo_dim),
        )
        self.geo_stats_encoder = nn.Sequential(
            nn.Linear(geo_stats_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, geo_dim),
        )

        global_dim = d_motion + d_mat + geo_dim
        self.global_context = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.spring_mod = nn.Linear(hidden_dim, 2)
        self.spring_base = nn.Sequential(
            nn.Linear(geo_dim + d_mat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.ctrl_spring_base = nn.Sequential(
            nn.Linear(geo_dim + d_mat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Start near the middle of the allowed log-k range; the class-prior
        # codebook and rollout gradients learn the scene-specific stiffness.
        nn.init.zeros_(self.spring_base[-1].weight)
        nn.init.zeros_(self.spring_base[-1].bias)
        nn.init.zeros_(self.ctrl_spring_base[-1].weight)
        nn.init.zeros_(self.ctrl_spring_base[-1].bias)
        nn.init.zeros_(self.spring_mod.weight)
        nn.init.zeros_(self.spring_mod.bias)
        self.damping_head = nn.Linear(hidden_dim, 2)
        self.contact_head = nn.Linear(hidden_dim, 5)

    def _global_hidden(
        self,
        pixel_values: torch.Tensor,
        material_dist: torch.Tensor,
        geo_stats: Optional[torch.Tensor],
    ):
        z_motion = self.motion_encoder(pixel_values)
        z_mat_part = self.material_codebook(material_dist)
        z_mat_global = z_mat_part.mean(dim=0)
        if geo_stats is None:
            z_geo_global = torch.zeros(
                self.geo_stats_encoder[-1].out_features,
                device=z_motion.device,
                dtype=z_motion.dtype,
            )
        else:
            z_geo_global = self.geo_stats_encoder(geo_stats.view(1, -1)).squeeze(0)
        h = self.global_context(torch.cat([z_motion, z_mat_global, z_geo_global], dim=-1))
        return h, z_mat_part

    def _spring_logits(self, geo_enc: torch.Tensor, z_mat: torch.Tensor, h_global: torch.Tensor,
                       head: nn.Module) -> torch.Tensor:
        raw = head(torch.cat([geo_enc, z_mat], dim=-1)).view(-1)
        scale_bias = self.spring_mod(h_global)
        scale = 1.0 + 0.25 * torch.tanh(scale_bias[0])
        bias = scale_bias[1]
        return raw * scale + bias

    def _logk_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        logk = math.log(self.logk_base) + self.logk_residual_scale * raw
        return _soft_clamp(
            logk,
            math.log(max(self.logk_min, 1e-6)),
            math.log(max(self.logk_max, self.logk_min + 1e-6)),
            self.logk_soft_clamp,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        z_geo: torch.Tensor,
        material_dist: torch.Tensor,
        edge_part_idx: torch.Tensor,
        geo_stats: Optional[torch.Tensor] = None,
        ctrl_rest_length: Optional[torch.Tensor] = None,
        ctrl_part_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        h_global, z_mat_part = self._global_hidden(pixel_values, material_dist, geo_stats)

        z_mat_edge = z_mat_part[edge_part_idx]
        edge_raw = self._spring_logits(
            self.geo_encoder(z_geo), z_mat_edge, h_global, self.spring_base,
        )

        d = self.damping_head(h_global)
        c = self.contact_head(h_global)
        result = {
            "log_k": self._logk_from_raw(edge_raw),
            "log_k_raw": edge_raw,
            "dashpot_damping": torch.sigmoid(d[0:1]) * 200.0,
            "drag_damping": torch.sigmoid(d[1:2]) * 20.0,
            "collide_elas": torch.sigmoid(c[0:1]),
            "collide_fric": torch.sigmoid(c[1:2]) * 2.0,
            "collide_object_elas": torch.sigmoid(c[2:3]),
            "collide_object_fric": torch.sigmoid(c[3:4]) * 2.0,
            "collision_dist": torch.sigmoid(c[4:5]) * 0.04 + 0.01,
        }

        if (
            ctrl_rest_length is not None and ctrl_rest_length.numel() > 0
            and ctrl_part_idx is not None and ctrl_part_idx.numel() > 0
        ):
            z_mat_ctrl = z_mat_part[ctrl_part_idx]
            ctrl_raw = self._spring_logits(
                self.rest_encoder(ctrl_rest_length.view(-1, 1)),
                z_mat_ctrl,
                h_global,
                self.ctrl_spring_base,
            )
            result["ctrl_log_k"] = self._logk_from_raw(ctrl_raw)
            result["ctrl_log_k_raw"] = ctrl_raw

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def forward_case(model: nn.Module, batch: dict, idx: int, device: torch.device,
                 pixel_values: torch.Tensor) -> dict:
    impl = _unwrap_model(model)
    ctrl_rest = batch["ctrl_rest_length"][idx].to(device)
    ctrl_rest = ctrl_rest.view(-1, 1) if ctrl_rest.numel() > 0 else None
    ctrl_part = batch.get("ctrl_part_idx")
    ctrl_part = ctrl_part[idx].to(device) if ctrl_part is not None and ctrl_rest is not None else None
    return impl(
        pixel_values=pixel_values,
        z_geo=batch["z_geo"][idx].to(device),
        material_dist=batch["material_dist"][idx].to(device),
        edge_part_idx=batch["edge_part_idx"][idx].to(device),
        geo_stats=batch["geo_stats"][idx].to(device) if "geo_stats" in batch else None,
        ctrl_rest_length=ctrl_rest,
        ctrl_part_idx=ctrl_part,
    )


def _init_runtime(case_name: str, train_frame: int, args) -> CaseRuntime:
    topo = {k: getattr(args, k) for k in (
        "use_knn_topology", "object_knn", "object_radius", "object_max_neighbours",
        "controller_radius", "controller_max_neighbours",
    )}
    return CaseRuntime(
        base_path=args.base_path,
        case_name=case_name,
        experiments_optimization_dir=args.experiments_optimization_dir,
        train_frame=train_frame,
        device=args.device,
        runtime_root=os.path.join("semantic", "runtime", f"material_simple_rank_{getattr(args, 'rank', 0)}"),
        topology_cfg=topo,
        gaussian_root=getattr(args, "gaussian_root", None),
    )


_GLOBAL_SCALES = {
    "collide_elas": 1.0,
    "collide_fric": 2.0,
    "collide_object_elas": 1.0,
    "collide_object_fric": 2.0,
    "collision_dist": 0.05,
    "dashpot_damping": 200.0,
    "drag_damping": 20.0,
}


def teacher_loss(model_out: dict, batch: dict, idx: int, device: torch.device, args) -> tuple[torch.Tensor, dict]:
    loss = torch.zeros((), device=device)
    stats = {"teacher_log_k": 0.0, "teacher_global": 0.0, "n_terms": 0}

    teacher_log_k = batch.get("teacher_log_k", [None])[idx]
    if teacher_log_k is not None and model_out["log_k"].numel() == teacher_log_k.numel():
        target = teacher_log_k.to(device).view(-1)
        pred = model_out["log_k"].view(-1)
        k_loss = F.smooth_l1_loss(pred / LOG_K_SPAN, target / LOG_K_SPAN)
        loss = loss + float(args.lambda_teacher_k) * k_loss
        stats["teacher_log_k"] = float(k_loss.detach().item())
        stats["n_terms"] += 1

    global_losses = []
    for key, scale in _GLOBAL_SCALES.items():
        teacher = batch.get(f"teacher_{key}", [None])[idx]
        if teacher is None or key not in model_out:
            continue
        pred = model_out[key].view(-1)
        target = teacher.to(device).view(-1)
        global_losses.append(F.smooth_l1_loss(pred / scale, target / scale))
    if global_losses:
        g_loss = torch.stack(global_losses).mean()
        loss = loss + float(args.lambda_teacher_global) * g_loss
        stats["teacher_global"] = float(g_loss.detach().item())
        stats["n_terms"] += len(global_losses)

    return loss, stats


def physics_prior_loss(model_out: dict, batch: dict, idx: int, device: torch.device, args) -> tuple[torch.Tensor, dict]:
    """Deprecated numeric VLM/GPT prior loss.

    The current training path intentionally does not regularize output physics
    parameters toward VLM-provided numeric means/sigmas. Material VLM output is
    consumed only as class probabilities through the learnable codebook.
    """
    zero = torch.zeros((), device=device)
    return zero, {"phys_part": 0.0, "phys_global": 0.0}


def _part_edge_moments(
    pred_logk_edges: torch.Tensor,
    edge_part_idx: torch.Tensor,
    num_parts: int,
    sigma_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = pred_logk_edges.view(-1, 1)
    part_idx = edge_part_idx.to(x.device).view(-1).long()
    counts = torch.zeros((num_parts, 1), device=x.device, dtype=x.dtype)
    sums = torch.zeros((num_parts, 1), device=x.device, dtype=x.dtype)
    counts.index_add_(0, part_idx, torch.ones_like(x))
    sums.index_add_(0, part_idx, x)
    mean = sums / counts.clamp_min(1.0)

    sq_sums = torch.zeros((num_parts, 1), device=x.device, dtype=x.dtype)
    centered_sq = (x - mean[part_idx]) ** 2
    sq_sums.index_add_(0, part_idx, centered_sq)
    var = sq_sums / (counts - 1.0).clamp_min(1.0)
    std = torch.sqrt(var + sigma_floor ** 2)
    return mean, std, counts


def part_physics_prior_barrier_loss(
    batch,
    idx: int,
    device: torch.device,
    pred_logk_edges: torch.Tensor,
    edge_part_idx: torch.Tensor,
    interval_sigma: float = 2.0,
    sigma_floor: float = 0.05,
    std_max_mult: float = 3.0,
    std_weight: float = 0.05,
) -> torch.Tensor:
    """Weakly penalize part moments only when they leave the LLM prior interval."""
    part_mu = batch["part_phys_prior_mu"][idx].to(device).view(-1, 1)
    part_log_sigma = batch["part_phys_prior_log_sigma"][idx].to(device).view(-1, 1)
    part_conf = batch["part_phys_prior_conf"][idx].to(device).view(-1, 1)
    if part_mu.numel() == 0 or float(part_conf.max().item()) <= 0.0:
        return torch.zeros((), device=device)

    sigma = part_log_sigma.exp().clamp_min(1e-4)
    mean, std, counts = _part_edge_moments(
        pred_logk_edges, edge_part_idx, int(part_mu.shape[0]), sigma_floor,
    )
    lower = part_mu - interval_sigma * sigma
    upper = part_mu + interval_sigma * sigma
    mean_violation = F.relu(lower - mean).square() + F.relu(mean - upper).square()
    mean_loss = mean_violation / sigma.square()

    max_std = (std_max_mult * sigma).clamp_min(sigma_floor)
    std_loss = F.relu(std - max_std).square() / max_std.square()

    valid = (counts > 0).to(mean.dtype)
    weights = part_conf * valid
    denom = weights.sum().clamp_min(1.0)
    return (weights * (mean_loss + std_weight * std_loss)).sum() / denom


def empirical_part_physics_prior_kl_loss(
    batch,
    idx: int,
    device: torch.device,
    pred_logk_edges: torch.Tensor,
    edge_part_idx: torch.Tensor,
    sigma_floor: float = 0.05,
) -> torch.Tensor:
    """KL(q_pred(part log_k) || p_llm(part log_k)) using per-part edge moments."""
    part_mu = batch["part_phys_prior_mu"][idx].to(device).view(-1, 1)
    part_log_sigma = batch["part_phys_prior_log_sigma"][idx].to(device).view(-1, 1)
    part_conf = batch["part_phys_prior_conf"][idx].to(device).view(-1, 1)
    if part_mu.numel() == 0 or float(part_conf.max().item()) <= 0.0:
        return torch.zeros((), device=device)

    num_parts = int(part_mu.shape[0])
    mean, sigma_q, counts = _part_edge_moments(
        pred_logk_edges, edge_part_idx, num_parts, sigma_floor,
    )
    sigma_p = part_log_sigma.exp().clamp_min(1e-4)
    kl = torch.log(sigma_p / sigma_q) + (
        sigma_q.square() + (mean - part_mu).square()
    ) / (2.0 * sigma_p.square()) - 0.5

    valid = (counts > 0).to(x.dtype)
    weights = part_conf * valid
    denom = weights.sum().clamp_min(1.0)
    return (weights * kl).sum() / denom


def global_physics_prior_barrier_loss(
    batch,
    idx: int,
    device: torch.device,
    model_out: Dict[str, torch.Tensor],
    interval_sigma: float = 2.0,
) -> torch.Tensor:
    if "global_phys_prior_mu" not in batch:
        return torch.zeros((), device=device)
    mu = batch["global_phys_prior_mu"][idx].to(device).view(1, -1)
    log_sigma = batch["global_phys_prior_log_sigma"][idx].to(device).view(1, -1)
    conf = batch["global_phys_prior_conf"][idx].to(device).view(1, 1)
    if mu.numel() == 0 or float(conf.max().item()) <= 0.0:
        return torch.zeros((), device=device)

    pred = torch.cat([
        model_out["collide_elas"].view(1, 1),
        model_out["collide_fric"].view(1, 1),
        model_out["collide_object_elas"].view(1, 1),
        model_out["collide_object_fric"].view(1, 1),
        model_out["collision_dist"].view(1, 1),
        model_out["dashpot_damping"].view(1, 1),
        model_out["drag_damping"].view(1, 1),
    ], dim=-1)
    sigma = log_sigma.exp().clamp_min(1e-4)
    lower = mu - interval_sigma * sigma
    upper = mu + interval_sigma * sigma
    violation = F.relu(lower - pred).square() + F.relu(pred - upper).square()
    return conf * (violation / sigma.square()).mean()


def training_loss(model_out: dict, batch: dict, idx: int, device: torch.device, args) -> tuple[torch.Tensor, dict]:
    prior, prior_stats = physics_prior_loss(model_out, batch, idx, device, args)
    teacher, teacher_stats = teacher_loss(model_out, batch, idx, device, args)
    loss = prior + teacher
    stats = {
        **prior_stats,
        "teacher_log_k": teacher_stats["teacher_log_k"],
        "teacher_global": teacher_stats["teacher_global"],
        "teacher_terms": teacher_stats["n_terms"],
    }
    return loss, stats


def _apply_model_out_to_sim(model_out: dict, model_logk: torch.Tensor, sim) -> None:
    sim.set_spring_Y(model_logk.detach())
    sim.set_collide(
        model_out["collide_elas"].detach().view(-1),
        model_out["collide_fric"].detach().view(-1),
    )
    sim.set_collide_object(
        model_out["collide_object_elas"].detach().view(-1),
        model_out["collide_object_fric"].detach().view(-1),
    )
    sim.collision_dist = float(model_out["collision_dist"].detach())
    # NOTE: assigning sim.dashpot_damping (Python attr) does NOT update the warp array
    # the kernels read from. Use set_dashpot_damping/set_drag_damping to actually
    # push the predicted value into the simulator.
    sim.set_dashpot_damping(model_out["dashpot_damping"].detach().view(-1))
    sim.set_drag_damping(model_out["drag_damping"].detach().view(-1))


def _build_model_logk(model_out: dict, runtime, sim, device: torch.device) -> Optional[torch.Tensor]:
    pred_logk = model_out["log_k"]
    if pred_logk.shape[0] != runtime.num_object_springs:
        return None
    obj_logk = _clamp_log_stiffness(pred_logk, runtime)
    ctrl_logk = model_out.get("ctrl_log_k")
    if ctrl_logk is not None and ctrl_logk.numel() > 0:
        ctrl_logk_sim = _clamp_log_stiffness(ctrl_logk, runtime)
    else:
        n_ctrl = sim.n_springs - runtime.num_object_springs
        ctrl_logk_sim = _clamp_log_stiffness(
            torch.full(
                (n_ctrl,),
                float(np.log(max(cfg.init_spring_Y, 1e-6))),
                device=device,
                dtype=obj_logk.dtype,
            ),
            runtime,
        )
    return torch.cat([obj_logk.view(-1), ctrl_logk_sim.view(-1)])


def _rollout_aux_loss(model_out: dict, batch: dict, idx: int, device: torch.device,
                      args, epoch: int) -> tuple[torch.Tensor, dict]:
    loss = torch.zeros((), device=device)
    stats = {
        "phys_part": 0.0,
        "phys_global": 0.0,
        "teacher_log_k": 0.0,
        "teacher_global": 0.0,
    }
    if float(args.lambda_teacher_k) > 0 or float(args.lambda_teacher_global) > 0:
        teacher, teacher_stats = teacher_loss(model_out, batch, idx, device, args)
        loss = loss + teacher
        stats["teacher_log_k"] = teacher_stats["teacher_log_k"]
        stats["teacher_global"] = teacher_stats["teacher_global"]
    return loss, stats


def train_epoch(model, train_loader, optimizer, device, runtimes: dict, args, epoch: int) -> dict:
    """Main training path: tracking/chamfer rollout loss, one network update per frame."""
    model.train()
    totals = dict(
        loss=0.0,
        track=0.0,
        geo=0.0,
        render=0.0,
        acc=0.0,
        phys_part=0.0,
        phys_global=0.0,
        teacher_log_k=0.0,
        teacher_global=0.0,
        n=0,
    )
    n_steps = sum(
        max(_resolve_train_frame(args, b["case_name"][i], int(b["train_frame"][i].item())) - 1, 1)
        for b in train_loader
        for i in range(len(b["case_name"]))
    )
    pbar = tqdm(total=max(n_steps, 1), desc="training", dynamic_ncols=True)
    lam_rnd = float(args.lambda_render)
    grad_scale = float(args.grad_scale)
    nan_kwargs = dict(nan=0.0, posinf=0.0, neginf=0.0)

    for batch in train_loader:
        for i, case_name in enumerate(batch["case_name"]):
            train_frame = _resolve_train_frame(args, case_name, int(batch["train_frame"][i].item()))
            pixel_values = load_video_frames(
                case_name,
                args.base_path,
                T=args.num_video_frames,
                image_size=args.videomae_image_size,
                device=device,
            )
            if pixel_values is None:
                pbar.update(max(train_frame - 1, 1))
                continue

            if case_name not in runtimes:
                runtimes[case_name] = _init_runtime(case_name, train_frame, args)
            runtime = runtimes[case_name]
            sim = runtime.sim

            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)
            cfg.acc_weight = float(args.lambda_acc_smooth)
            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
            sim.set_acc_count(False)

            K_cam = batch["cam0_intrinsics"][i].to(device)
            w2c = batch["cam0_w2c"][i].to(device)
            wh = batch["wh"][i].to(device)
            W, H = int(wh[0].item()), int(wh[1].item())

            tr = geo = rnd = 0.0
            acc_total = 0.0
            phys_part = phys_global = teacher_k = teacher_global = 0.0
            steps = 0
            invalid_case = False

            for frame_idx in range(1, train_frame):
                optimizer.zero_grad(set_to_none=True)

                model_out = forward_case(model, batch, i, device, pixel_values)
                model_logk = _build_model_logk(model_out, runtime, sim, device)
                if model_logk is None:
                    pbar.write(
                        f"[skip] {case_name}: pred_logk={tuple(model_out['log_k'].shape)} "
                        f"runtime_object_springs={runtime.num_object_springs} "
                        f"sim_springs={sim.n_springs}"
                    )
                    invalid_case = True
                    pbar.update(max(train_frame - frame_idx, 1))
                    break

                _apply_model_out_to_sim(model_out, model_logk, sim)
                sim.set_controller_target(frame_idx)
                if sim.object_collision_flag:
                    sim.update_collision_graph()
                with sim.tape:
                    sim.step()
                    sim.calculate_loss()

                tr += wp.to_torch(sim.track_loss, requires_grad=False).item()
                geo += wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                acc_step = float(wp.to_torch(sim.acc_loss, requires_grad=False).item())

                pred_pts = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=True)
                rnd_val = torch.zeros((), device=device)
                if runtime.gaussians is not None and lam_rnd > 0:
                    gt_color = runtime.load_gt_color_cam0(frame_idx, W, H)
                    if gt_color is not None:
                        view = _make_gs_view(w2c, K_cam, H, W, device)
                        fg_mask = runtime.load_union_mask_cam0(frame_idx, W, H)
                        rnd_val = gaussian_render_l1_loss(
                            pred_pts,
                            runtime.sim_init_pts,
                            runtime.gaussians,
                            runtime.knn_indices,
                            runtime.knn_weights_vals,
                            runtime.gs_init_xyz,
                            view,
                            gt_color,
                            runtime.gs_background,
                            mask=fg_mask,
                        )
                rnd += float(rnd_val.item())

                if rnd_val.requires_grad and lam_rnd > 0:
                    render_grad = torch.autograd.grad(rnd_val, pred_pts)[0] * lam_rnd
                    sim.tape.backward(
                        sim.loss,
                        grads={sim.wp_states[-1].wp_x: wp.from_torch(render_grad.contiguous())},
                    )
                else:
                    sim.tape.backward(sim.loss)

                g_logk = torch.nan_to_num(
                    wp.to_torch(sim.wp_spring_Y.grad, requires_grad=False).detach() * grad_scale,
                    **nan_kwargs,
                )
                grads = [
                    g_logk,
                    torch.nan_to_num(wp.to_torch(sim.wp_collide_elas.grad, requires_grad=False).detach(), **nan_kwargs),
                    torch.nan_to_num(wp.to_torch(sim.wp_collide_fric.grad, requires_grad=False).detach(), **nan_kwargs),
                    torch.nan_to_num(wp.to_torch(sim.wp_collide_object_elas.grad, requires_grad=False).detach(), **nan_kwargs),
                    torch.nan_to_num(wp.to_torch(sim.wp_collide_object_fric.grad, requires_grad=False).detach(), **nan_kwargs),
                    torch.nan_to_num(wp.to_torch(sim.wp_dashpot_damping.grad, requires_grad=False).detach(), **nan_kwargs),
                    torch.nan_to_num(wp.to_torch(sim.wp_drag_damping.grad, requires_grad=False).detach(), **nan_kwargs),
                ]

                bad_sim_grad = not all(_is_finite_tensor(g) for g in grads)
                if not bad_sim_grad:
                    aux, aux_stats = _rollout_aux_loss(model_out, batch, i, device, args, epoch)
                    if aux.requires_grad:
                        aux.backward(retain_graph=True)
                    phys_part += aux_stats["phys_part"]
                    phys_global += aux_stats["phys_global"]
                    teacher_k += aux_stats["teacher_log_k"]
                    teacher_global += aux_stats["teacher_global"]

                    bwd_tensors = [model_logk]
                    bwd_grads = [grads[0]]
                    for tensor, grad in (
                        (model_out["collide_elas"], grads[1]),
                        (model_out["collide_fric"], grads[2]),
                        (model_out["collide_object_elas"], grads[3]),
                        (model_out["collide_object_fric"], grads[4]),
                        (model_out["dashpot_damping"], grads[5]),
                        (model_out["drag_damping"], grads[6]),
                    ):
                        if tensor.requires_grad:
                            bwd_tensors.append(tensor)
                            bwd_grads.append(grad)
                    # Damping barrier: soft floor on predicted dashpot/drag,
                    # pushing damping up to avoid under-damped (bouncy) solutions.
                    # Penalty is unit-normalized: (relu((floor - x) / floor))^2 in [0, 1].
                    lam_db = float(args.lambda_damp_barrier)
                    if lam_db > 0:
                        d_floor = max(float(args.min_dashpot), 1e-6)
                        r_floor = max(float(args.min_drag), 1e-6)
                        d_pen = torch.relu((d_floor - model_out["dashpot_damping"]) / d_floor).pow(2).sum()
                        r_pen = torch.relu((r_floor - model_out["drag_damping"]) / r_floor).pow(2).sum()
                        damp_barrier = lam_db * (d_pen + r_pen)
                        if damp_barrier.requires_grad:
                            damp_barrier.backward(retain_graph=True)
                    torch.autograd.backward(bwd_tensors, bwd_grads)

                bad_param_grad = (
                    bad_sim_grad
                    or any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                )
                if bad_param_grad:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    optimizer.step()

                # Activate jerk smoothness from the second frame onward and refresh
                # prev_acc with the current step's (v_t - v_0). Mirrors cma_optimize_warp.
                if float(args.lambda_acc_smooth) > 0:
                    if int(wp.to_torch(sim.acc_count, requires_grad=False)[0].item()) == 0:
                        sim.set_acc_count(True)
                    sim.update_acc()
                acc_total += acc_step

                sim.tape.reset()
                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
                steps += 1
                pbar.update(1)

            if invalid_case or steps == 0:
                if steps == 0 and not invalid_case:
                    pbar.update(1)
                continue

            mt, mg, mr = tr / steps, geo / steps, rnd / steps
            mac = acc_total / steps
            if not np.isfinite(mt + mg + mr):
                continue
            total = float(args.lambda_track) * mt + float(args.lambda_geo) * mg + lam_rnd * mr
            totals["loss"] += total
            totals["track"] += mt
            totals["geo"] += mg
            totals["render"] += mr
            totals["acc"] += mac
            totals["phys_part"] += phys_part / steps
            totals["phys_global"] += phys_global / steps
            totals["teacher_log_k"] += teacher_k / steps
            totals["teacher_global"] += teacher_global / steps
            totals["n"] += 1
            pbar.set_postfix(total=f"{total:.4e}", track=f"{mt:.3e}", geo=f"{mg:.3e}")

    pbar.close()
    n = max(totals["n"], 1)
    keys = ("loss", "track", "geo", "render", "acc", "phys_part", "phys_global", "teacher_log_k", "teacher_global")
    out = {k: totals[k] / n for k in keys}
    out["num_graphs"] = totals["n"]
    if _is_distributed():
        dev = next(model.parameters()).device
        for k in keys:
            out[k] = _all_reduce_mean(totals[k], totals["n"], device=dev)
    return out


@torch.no_grad()
def evaluate(model, loader, device, runtimes: dict, args, vis_dir: Optional[str] = None, epoch: int = 0) -> dict:
    model.eval()
    totals = dict(loss=0.0, track=0.0, geo=0.0, render=0.0, n=0)
    lam_rnd = float(args.lambda_render)
    for batch in loader:
        for i, case_name in enumerate(batch["case_name"]):
            train_frame = _resolve_train_frame(args, case_name, int(batch["train_frame"][i].item()))
            pixel_values = load_video_frames(
                case_name,
                args.base_path,
                T=args.num_video_frames,
                image_size=args.videomae_image_size,
                device=device,
            )
            if pixel_values is None:
                continue
            if case_name not in runtimes:
                runtimes[case_name] = _init_runtime(case_name, train_frame, args)
            runtime = runtimes[case_name]
            sim = runtime.sim
            model_out = forward_case(model, batch, i, device, pixel_values)
            model_logk = _build_model_logk(model_out, runtime, sim, device)
            if model_logk is None:
                continue

            _apply_model_out_to_sim(model_out, model_logk, sim)
            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)

            K_cam = batch["cam0_intrinsics"][i].to(device)
            w2c = batch["cam0_w2c"][i].to(device)
            wh = batch["wh"][i].to(device)
            W, H = int(wh[0].item()), int(wh[1].item())

            tr = geo = rnd = 0.0
            steps = 0
            verts = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()] if vis_dir else None
            for frame_idx in range(1, train_frame):
                sim.set_controller_target(frame_idx)
                if sim.object_collision_flag:
                    sim.update_collision_graph()
                sim.step()
                sim.calculate_loss()
                tr += wp.to_torch(sim.track_loss, requires_grad=False).item()
                geo += wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                if runtime.gaussians is not None and lam_rnd > 0:
                    gt_color = runtime.load_gt_color_cam0(frame_idx, W, H)
                    if gt_color is not None:
                        pts = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
                        view = _make_gs_view(w2c, K_cam, H, W, device)
                        fg_mask = runtime.load_union_mask_cam0(frame_idx, W, H)
                        rnd_val = gaussian_render_l1_loss(
                            pts,
                            runtime.sim_init_pts,
                            runtime.gaussians,
                            runtime.knn_indices,
                            runtime.knn_weights_vals,
                            runtime.gs_init_xyz,
                            view,
                            gt_color,
                            runtime.gs_background,
                            mask=fg_mask,
                        )
                        rnd += float(rnd_val.item())
                if verts is not None:
                    verts.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu())
                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
                steps += 1

            if steps > 0:
                mt, mg, mr = tr / steps, geo / steps, rnd / steps
                totals["track"] += mt
                totals["geo"] += mg
                totals["render"] += mr
                totals["loss"] += float(args.lambda_track) * mt + float(args.lambda_geo) * mg + lam_rnd * mr
                totals["n"] += 1

                if vis_dir is not None and verts is not None:
                    vid_path = os.path.join(vis_dir, f"sim_inference_{case_name}_epoch{epoch:04d}.mp4")
                    try:
                        vertices = torch.stack(verts)
                        visualize_pc(
                            vertices[:, :runtime.trainer.num_all_points, :],
                            runtime.trainer.object_colors,
                            runtime.trainer.controller_points,
                            visualize=False, save_video=True, save_path=vid_path,
                            width=W, height=H,
                            intrinsic=K_cam.detach().cpu().numpy(),
                            w2c=w2c.detach().cpu().numpy(),
                            overlay_path=os.path.join(args.base_path, case_name, "color"),
                        )
                        print(f"  [vis] Saved {vid_path}")
                    except Exception as e:
                        print(f"  [vis] Failed {vid_path}: {e}")

    n = max(totals["n"], 1)
    out = {k: totals[k] / n for k in ("loss", "track", "geo", "render")}
    out["num_graphs"] = totals["n"]
    return out


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", type=str, default="/home/yyang/mnt/data/PhysTwin")
    p.add_argument("--experiments_dir", type=str, default="/home/yyang/mnt/data/PhysTwin")
    p.add_argument("--experiments_optimization_dir", type=str, default="/home/yyang/mnt/data/PhysTwin")
    p.add_argument("--case_to_material", type=str, default="semantic/case_to_material_different_types.json")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--sem_cache_dir", type=str, default="semantic/cache")
    p.add_argument("--gaussian_root", type=str, default="gaussian_output")
    p.add_argument("--save_dir", type=str, default="checkpoints/video_material_simple")
    p.add_argument("--case_name", type=str, default="")
    p.add_argument("--train_ratio", type=float, default=0.8)

    p.add_argument("--videomae_model", type=str, default="MCG-NJU/videomae-base")
    p.add_argument("--videomae_image_size", type=int, default=224)
    p.add_argument("--num_video_frames", type=int, default=16)
    p.add_argument("--d_motion", type=int, default=128)
    p.add_argument("--mat_codebook_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_materials", type=int, default=10)
    p.add_argument("--logk_base", type=float, default=3e4)
    p.add_argument("--logk_min", type=float, default=1e3)
    p.add_argument("--logk_max", type=float, default=1e5)
    p.add_argument("--logk_residual_scale", type=float, default=1.0)
    p.add_argument("--logk_soft_clamp", type=float, default=0.25)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--save_best_only", action="store_true",
                   help="Save only last_checkpoint.pth + best_checkpoint.pth (by test loss) instead of every-10-epoch dumps")
    p.add_argument("--vis_every", type=int, default=0,
                   help="Run simulation rollout visualization every N epochs (0 = disabled)")

    p.add_argument("--lambda_track", type=float, default=1.0)
    p.add_argument("--lambda_geo", type=float, default=1.0)
    p.add_argument("--lambda_render", type=float, default=0.0)
    p.add_argument("--grad_scale", type=float, default=1000.0)
    p.add_argument("--lambda_phys_prior", type=float, default=0.0,
                   help="Deprecated and ignored; numeric VLM/GPT physics prior loss is disabled")
    p.add_argument("--phys_prior_start_epoch", type=int, default=5)
    p.add_argument("--phys_prior_part_mode", choices=("barrier", "empirical_kl", "mean_nll"), default="barrier")
    p.add_argument("--phys_prior_global_mode", choices=("barrier", "nll"), default="barrier")
    p.add_argument("--phys_prior_interval_sigma", type=float, default=2.0)
    p.add_argument("--phys_prior_std_max_mult", type=float, default=3.0)
    p.add_argument("--phys_prior_std_weight", type=float, default=0.05)
    p.add_argument("--pred_part_sigma_floor", type=float, default=0.05)
    p.add_argument("--lambda_teacher_k", type=float, default=0.0)
    p.add_argument("--lambda_teacher_global", type=float, default=0.0)
    # --- Anti-vibration regularizers ---
    p.add_argument("--lambda_acc_smooth", type=float, default=0.0,
                   help="Warp-side jerk smoothness weight (cfg.acc_weight). Smooth-L1 on per-node "
                        "(v_t - v_{t-1}) - (v_{t-1} - v_{t-2}). Adds gradient to spring_Y AND damping.")
    p.add_argument("--lambda_damp_barrier", type=float, default=0.0,
                   help="Torch-side soft barrier on predicted dashpot/drag below configured floors.")
    p.add_argument("--min_dashpot", type=float, default=30.0,
                   help="Soft floor for predicted dashpot_damping (range 0-200). Penalty (min - x)^2 below.")
    p.add_argument("--min_drag", type=float, default=3.0,
                   help="Soft floor for predicted drag_damping (range 0-20).")
    p.add_argument("--fit_all_frames", action="store_true",
                   help="Override per-batch train_frame to split.json frame_len so the trainer "
                        "fits test frames in addition to the train split.")

    p.add_argument("--use_knn_topology", action="store_true")
    p.add_argument("--object_knn", type=int, default=30)
    p.add_argument("--object_radius", type=float, default=0.02)
    p.add_argument("--object_max_neighbours", type=int, default=30)
    p.add_argument("--controller_radius", type=float, default=0.04)
    p.add_argument("--controller_max_neighbours", type=int, default=50)
    p.add_argument("--spring_y_min", type=float, default=None)
    p.add_argument("--spring_y_max", type=float, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
        rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.rank = rank
    set_all_seeds(args.seed + rank)
    if args.spring_y_min is not None:
        cfg.spring_Y_min = float(args.spring_y_min)
    if args.spring_y_max is not None:
        cfg.spring_Y_max = float(args.spring_y_max)
    os.makedirs(args.save_dir, exist_ok=True)
    log_fh = open(os.path.join(args.save_dir, "train.log"), "a") if rank == 0 else None

    def log(msg: str):
        if rank == 0:
            print(msg)
            print(msg, file=log_fh, flush=True)

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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
        distributed=is_dist,
        rank=rank,
        world_size=dist.get_world_size() if is_dist else 1,
    )

    if args.case_name:
        from torch.utils.data import DataLoader, Subset
        from material_param_dataset import collate_graph_batch

        matched = [s for s in full_dataset.samples if s["case_name"] == args.case_name]
        if not matched:
            raise ValueError(f"case_name={args.case_name!r} not found in dataset")
        single_ds = Subset(full_dataset, [full_dataset.samples.index(matched[0])])
        train_loader = DataLoader(single_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_graph_batch)
        test_loader = DataLoader(single_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_graph_batch)

    model = SimpleVideoMaterialPhysicsModel(
        videomae_model=args.videomae_model,
        d_motion=args.d_motion,
        d_mat=args.mat_codebook_dim,
        hidden_dim=args.hidden_dim,
        num_materials=args.num_materials,
        logk_base=args.logk_base,
        logk_min=args.logk_min,
        logk_max=args.logk_max,
        logk_residual_scale=args.logk_residual_scale,
        logk_soft_clamp=args.logk_soft_clamp,
    ).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        ckpt_path = _find_latest_checkpoint(args.save_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            log(f"[resume] {ckpt_path}, epoch {start_epoch + 1}")

    log(
        f"SimpleVideoMaterialPhysicsModel trainable_params={_unwrap_model(model).count_parameters():,} "
        f"frozen_backbone=True d_motion={args.d_motion} d_mat={args.mat_codebook_dim} "
        f"main_loss=tracking+geo numeric_phys_prior=disabled "
        f"class_prior=material_dist@learnable_codebook "
        f"logk=absolute_head(base={args.logk_base:g}, min={args.logk_min:g}, max={args.logk_max:g}, "
        f"scale={args.logk_residual_scale:g}, "
        f"soft_clamp={args.logk_soft_clamp:g})"
    )

    train_runtimes = {}
    eval_runtimes = {}
    best_test_loss = float("inf")
    vis_dir = None
    if rank == 0 and args.vis_every > 0:
        vis_dir = os.path.join(args.save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        if is_dist:
            train_loader.sampler.set_epoch(epoch)
        stats = train_epoch(model, train_loader, optimizer, device, train_runtimes, args, epoch)
        scheduler.step()
        log(
            f"[Epoch {epoch+1:04d}] Train: loss={stats['loss']:.6f} "
            f"track={stats['track']:.6f} geo={stats['geo']:.6f} render={stats['render']:.6f} "
            f"acc={stats['acc']:.6e} "
            f"phys_part={stats['phys_part']:.6f} phys_global={stats['phys_global']:.6f} "
            f"teacher_k={stats['teacher_log_k']:.6f} teacher_global={stats['teacher_global']:.6f} "
            f"graphs={stats['num_graphs']}"
        )
        ev = None
        if (epoch + 1) % args.eval_every == 0:
            do_vis = vis_dir is not None and (epoch + 1) % args.vis_every == 0
            ev = evaluate(
                model, test_loader, device, eval_runtimes, args,
                vis_dir=vis_dir if do_vis else None,
                epoch=epoch + 1,
            )
            log(
                f"[Epoch {epoch+1:04d}] Test: loss={ev['loss']:.6f} "
                f"track={ev['track']:.6f} geo={ev['geo']:.6f} render={ev['render']:.6f} "
                f"graphs={ev['num_graphs']}"
            )
        if rank == 0 and (epoch + 1) % args.eval_every == 0:
            payload = {
                "epoch": epoch + 1,
                "model_state_dict": _unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }
            if args.save_best_only:
                last_path = os.path.join(args.save_dir, "last_checkpoint.pth")
                torch.save(payload, last_path)
                log(f"Saved last checkpoint to: {last_path}")
                if ev is not None and ev["loss"] < best_test_loss:
                    best_test_loss = ev["loss"]
                    best_path = os.path.join(args.save_dir, "best_checkpoint.pth")
                    torch.save(payload, best_path)
                    log(f"Saved best checkpoint (test_loss={ev['loss']:.6f}) to: {best_path}")
            else:
                ckpt = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
                torch.save(payload, ckpt)
                log(f"Saved checkpoint to: {ckpt}")

    if log_fh:
        log_fh.close()
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
