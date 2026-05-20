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
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import knn_weights_sparse
from gaussian_splatting.utils.graphics_utils import focal2fov
from gs_render import remove_gaussians_with_low_opacity


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def load_video_frames(case_name: str, base_path: str, T: int = 16,
                      image_size: int = 224, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Load T uniformly sampled RGB frames as [1, T, C, H, W] normalized tensor.

    Returns None if frames are not available (caller handles fallback).
    """
    from torchvision import transforms
    from PIL import Image

    color_dir = os.path.join(base_path, case_name, "color", "0")
    if not os.path.isdir(color_dir):
        return None
    frame_files = sorted(f for f in os.listdir(color_dir) if f.endswith(".png"))
    if not frame_files:
        return None

    idxs = np.linspace(0, len(frame_files) - 1, T, dtype=int)
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    frames = [tfm(Image.open(os.path.join(color_dir, frame_files[i])).convert("RGB")) for i in idxs]
    t = torch.stack(frames, dim=0).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Return the highest numbered epoch checkpoint from save_dir."""
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth")
    checkpoint_paths = glob.glob(pattern)
    if not checkpoint_paths:
        return None

    def _extract_epoch(path: str) -> int:
        match = re.search(r"checkpoint_epoch_(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    checkpoint_paths.sort(key=_extract_epoch)
    return checkpoint_paths[-1]


def _load_case_cfg(
    base_path: str,
    case_name: str,
    experiments_optimization_dir: str,
    topology_cfg: Optional[Dict[str, float]] = None,
) -> None:
    """Load case-specific simulator configuration without optimized teacher parameters."""
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    cfg.use_edge_gating = False
    cfg.sem_cache_dir = "__disabled__"
    topology_cfg = topology_cfg or {}
    optimal_path = os.path.join(experiments_optimization_dir, case_name, "optimal_params.pkl")
    if os.path.exists(optimal_path):
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        for key in (
            "object_radius",
            "object_max_neighbours",
            "controller_radius",
            "controller_max_neighbours",
        ):
            if key in optimal_params:
                topology_cfg[key] = optimal_params[key]
    cfg.use_knn_topology = bool(topology_cfg.get("use_knn_topology", False))
    cfg.object_knn = int(topology_cfg.get("object_knn", getattr(cfg, "object_knn", 30)))
    cfg.object_radius = float(topology_cfg.get("object_radius", getattr(cfg, "object_radius", 0.02)))
    cfg.object_max_neighbours = int(topology_cfg.get("object_max_neighbours", getattr(cfg, "object_max_neighbours", 30)))
    cfg.controller_radius = float(topology_cfg.get("controller_radius", getattr(cfg, "controller_radius", 0.04)))
    cfg.controller_max_neighbours = int(topology_cfg.get("controller_max_neighbours", getattr(cfg, "controller_max_neighbours", 50)))

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


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _all_reduce_mean(value: float, count: float = 1.0, device: Optional[torch.device] = None) -> float:
    if not _is_distributed():
        return value / max(count, 1.0)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([value, count], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(float(tensor[1].item()), 1.0)
    return float(tensor[0].item() / total_count)


def setup_distributed(args) -> Tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
    return rank, local_rank, world_size, distributed


def cleanup_distributed() -> None:
    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


class CaseRuntime:
    """Runtime environment for a single case with physics simulation."""

    def __init__(
        self,
        base_path: str,
        case_name: str,
        experiments_optimization_dir: str,
        train_frame: int,
        device: str,
        runtime_root: str = "semantic/runtime",
        topology_cfg: Optional[Dict[str, float]] = None,
        gaussian_root: str = None,
    ):
        _load_case_cfg(base_path, case_name, experiments_optimization_dir, topology_cfg=topology_cfg)
        wp.set_device(device)
        cfg.device = device
        self.case_name = case_name
        self.case_dir = os.path.join(base_path, case_name)
        self.train_frame = int(train_frame)
        self.device = device
        self.color_cache: Dict[int, torch.Tensor] = {}

        self.trainer = InvPhyTrainerWarp(
            data_path=os.path.join(base_path, case_name, "final_data.pkl"),
            base_dir=os.path.join(runtime_root, case_name),
            train_frame=self.train_frame,
            pure_inference_mode=True,
            device=device,
        )
        self.sim = self.trainer.simulator
        self.num_object_springs = self.trainer.num_object_springs
        self.num_original_points = self.trainer.num_original_points

        # Initial simulation particle positions (for computing motion in render loss)
        sp = self.trainer.structure_points
        if isinstance(sp, np.ndarray):
            self.sim_init_pts = torch.from_numpy(sp).float().to(device)
        else:
            self.sim_init_pts = sp.float().to(device)

        # Gaussian model for re-rendering loss (optional)
        self.gaussians = None
        self.gs_init_xyz = None
        self.knn_indices = None
        self.knn_weights_vals = None
        self.gs_background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        if gaussian_root is not None:
            ply_candidates = sorted(glob.glob(os.path.join(
                gaussian_root, case_name, "*", "point_cloud", "iteration_*", "point_cloud.ply"
            )))
            if ply_candidates:
                gs_path = ply_candidates[-1]
                self.gaussians = GaussianModel(sh_degree=3)
                self.gaussians.load_ply(gs_path)
                self.gaussians = remove_gaussians_with_low_opacity(self.gaussians, 0.1)
                self.gaussians.isotropic = self.gaussians._scaling.shape[1] == 1
                self.gs_init_xyz = self.gaussians.get_xyz.detach().clone()
                self.knn_weights_vals, self.knn_indices = knn_weights_sparse(
                    self.sim_init_pts, self.gs_init_xyz, K=16
                )
            else:
                print(f"[warn] No Gaussian model found for {case_name} under {gaussian_root}")

    def load_gt_color_cam0(self, frame_idx: int, width: int, height: int) -> Optional[torch.Tensor]:
        """Load GT color frame for cam0. Returns [3, H, W] float32 tensor in [0,1], or None."""
        if frame_idx in self.color_cache:
            return self.color_cache[frame_idx]

        img_path = os.path.join(self.case_dir, "color", "0", f"{frame_idx}.png")
        if not os.path.isfile(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        self.color_cache[frame_idx] = t
        return t

    def load_union_mask_cam0(self, frame_idx: int, width: int, height: int) -> torch.Tensor:
        """Load union of all object masks for a frame. Returns [1, 1, H, W] float32."""
        mask_dir = os.path.join(self.case_dir, "mask", "0")
        frame_name = f"{frame_idx}.png"
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*", frame_name)))
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
        return torch.from_numpy(union_mask)[None, None].to(self.device)


def _make_gs_view(w2c: torch.Tensor, K: torch.Tensor, height: int, width: int, device: str) -> Camera:
    w2c_np = w2c.cpu().numpy() if torch.is_tensor(w2c) else w2c
    K_np = K.cpu().numpy() if torch.is_tensor(K) else K
    R = np.transpose(w2c_np[:3, :3])
    T = w2c_np[:3, 3]
    FovY = focal2fov(float(K_np[1, 1]), height)
    FovX = focal2fov(float(K_np[0, 0]), width)
    K_t = K if torch.is_tensor(K) else torch.tensor(K, dtype=torch.float32, device=device)
    return Camera(
        (width, height), colmap_id="0", R=R, T=T, FoVx=FovX, FoVy=FovY,
        depth_params=None, image=None, invdepthmap=None, image_name="0", uid="0",
        data_device=device, train_test_exp=None, is_test_dataset=None, is_test_view=None,
        K=K_t, normal=None, depth=None, occ_mask=None,
    )


def gaussian_render_l1_loss(
    pred_pts: torch.Tensor,
    init_pts: torch.Tensor,
    gaussians: GaussianModel,
    knn_indices: torch.Tensor,
    knn_weights_vals: torch.Tensor,
    gs_init_xyz: torch.Tensor,
    view: Camera,
    gt_color: torch.Tensor,
    background: torch.Tensor,
    mask: torch.Tensor = None,
    return_rendered: bool = False,
):
    """L1 re-rendering loss via KNN-weighted Gaussian deformation.

    mask: [1, 1, H, W] float in [0, 1] — object foreground mask.
          If provided, GT background is replaced with white before L1.
    return_rendered: if True, returns (loss, rendered_rgb [3,H,W]).
    """
    motion = pred_pts - init_pts                                                   # [N_pts, 3]
    k_motions = motion[knn_indices]                                                # [N_gs, K, 3]
    new_xyz = gs_init_xyz + (k_motions * knn_weights_vals.unsqueeze(-1)).sum(1)   # [N_gs, 3]
    gaussians._xyz = new_xyz
    rendered = render_gaussian(view, gaussians, None, background)["render"]
    if rendered.shape[0] == 4:
        rendered = rendered[:3]
    if mask is not None:
        # Replace GT background with white so background matches the render
        fg = mask[0, 0:1]                                              # [1, H, W] float
        gt_color = gt_color * fg + (1.0 - fg)                         # background → white
    loss = F.l1_loss(rendered, gt_color)
    if return_rendered:
        return loss, rendered.detach(), gt_color.detach()
    return loss



def forward_model(
    model: nn.Module,
    model_type: str,
    z_geo: torch.Tensor,
    z_sem: torch.Tensor,
    material_dist: torch.Tensor,
    edge_part_idx: torch.Tensor,
    z_sem_global: torch.Tensor = None,
    part_features: torch.Tensor = None,
    prior_source: str = "vlm",
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
            z_geo,
            z_sem,
            material_dist,
            edge_part_idx,
            z_sem_global,
            part_features=part_features,
            prior_source=prior_source,
            ctrl_sem=ctrl_sem,
            ctrl_rest_length=ctrl_rest_length,
            ctrl_part_idx=ctrl_part_idx,
        )
    elif model_type == "part_level":
        return model(
            part_features,
            material_dist,
            edge_part_idx,
            z_sem_global,
            prior_source=prior_source,
            ctrl_sem=ctrl_sem,
            ctrl_rest_length=ctrl_rest_length,
            ctrl_part_idx=ctrl_part_idx,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _build_slices(lengths: List[int]) -> List[slice]:
    slices: List[slice] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        slices.append(slice(start, end))
        start = end
    return slices


def forward_model_batched(
    model: nn.Module,
    model_type: str,
    batch,
    device: torch.device,
    prior_source: str = "vlm",
) -> List[Dict[str, torch.Tensor]]:
    """Run one batched model forward and split outputs back per case."""
    num_cases = len(batch["case_name"])
    if num_cases == 0:
        return []

    if model_type == "part_level":
        material_parts = [batch["material_dist"][i].to(device) for i in range(num_cases)]
        part_feature_parts = [batch["part_features"][i].to(device) for i in range(num_cases)]
        z_sem_global_parts = [batch["z_sem_global"][i].to(device).view(1, -1) for i in range(num_cases)]
        ctrl_sem_parts = [batch["ctrl_sem"][i].to(device) for i in range(num_cases)]
        ctrl_rest_parts = [batch["ctrl_rest_length"][i].to(device) for i in range(num_cases)]

        edge_lengths = [int(batch["edge_part_idx"][i].numel()) for i in range(num_cases)]
        part_lengths = [int(material.shape[0]) for material in material_parts]
        ctrl_lengths = [int(part.shape[0]) for part in ctrl_sem_parts]
        edge_slices = _build_slices(edge_lengths)
        ctrl_slices = _build_slices(ctrl_lengths)
        part_slices = _build_slices(part_lengths)
        part_offsets = [part_slice.start for part_slice in part_slices]

        material_cat = torch.cat(material_parts, dim=0)
        part_features_cat = torch.cat(part_feature_parts, dim=0)
        edge_part_idx_cat = torch.cat(
            [batch["edge_part_idx"][i].to(device) + part_offsets[i] for i in range(num_cases)],
            dim=0,
        )
        model_impl = _unwrap_model(model)
        z_mat_parts_cat = model_impl.encode_material_prior(
            material_cat,
            part_features_cat,
            prior_source=prior_source,
        )
        raw_log_k_per_part_cat = model_impl.physics_head(part_features_cat, z_mat_parts_cat)
        log_k_min = float(np.log(1e3))
        log_k_max = float(np.log(1e5)) # ？
        log_k_per_part_cat = torch.sigmoid(raw_log_k_per_part_cat) * (log_k_max - log_k_min) + log_k_min
        log_k_cat = log_k_per_part_cat[edge_part_idx_cat]

        ctrl_log_k_cat = None
        if sum(ctrl_lengths) > 0:
            ctrl_sem_cat = torch.cat(ctrl_sem_parts, dim=0)
            ctrl_rest_cat = torch.cat(ctrl_rest_parts, dim=0)
            ctrl_part_idx_cat = torch.cat(
                [batch["ctrl_part_idx"][i].to(device) + part_offsets[i] for i in range(num_cases)],
                dim=0,
            )
            ctrl_z_mat_cat = z_mat_parts_cat[ctrl_part_idx_cat]
            raw_ctrl_log_k_cat = model.ctrl_decoder(ctrl_sem_cat, ctrl_z_mat_cat, ctrl_rest_cat)
            ctrl_log_k_cat = torch.sigmoid(raw_ctrl_log_k_cat) * (log_k_max - log_k_min) + log_k_min

        z_sem_global_cat = torch.cat(z_sem_global_parts, dim=0)
        case_outputs = []
        for idx in range(num_cases):
            part_slice = part_slices[idx]
            global_out = model.global_decoder(z_sem_global_cat[idx:idx + 1], z_mat_parts_cat[part_slice])
            case_out = {
                "log_k": log_k_cat[edge_slices[idx]],
                "collide_elas": global_out["collide_elas"],
                "collide_fric": global_out["collide_fric"],
                "collide_object_elas": global_out["collide_object_elas"],
                "collide_object_fric": global_out["collide_object_fric"],
                "collision_dist": global_out["collision_dist"],
                "dashpot_damping": global_out["dashpot_damping"],
                "drag_damping": global_out["drag_damping"],
            }
            if ctrl_log_k_cat is not None and ctrl_lengths[idx] > 0:
                case_out["ctrl_log_k"] = ctrl_log_k_cat[ctrl_slices[idx]]
            case_outputs.append(case_out)
        return case_outputs

    if model_type != "edge_level":
        raise ValueError(f"Unknown model_type: {model_type}")

    material_parts = [batch["material_dist"][i].to(device) for i in range(num_cases)]
    z_geo_parts = [batch["z_geo"][i].to(device) for i in range(num_cases)]
    z_sem_parts = [batch["z_sem"][i].to(device) for i in range(num_cases)]
    z_sem_global_parts = [batch["z_sem_global"][i].to(device).view(1, -1) for i in range(num_cases)]
    part_feature_parts = [batch["part_features"][i].to(device) for i in range(num_cases)]
    ctrl_sem_parts = [batch["ctrl_sem"][i].to(device) for i in range(num_cases)]
    ctrl_rest_parts = [batch["ctrl_rest_length"][i].to(device) for i in range(num_cases)]

    edge_lengths = [int(part.shape[0]) for part in z_geo_parts]
    part_lengths = [int(part.shape[0]) for part in material_parts]
    ctrl_lengths = [int(part.shape[0]) for part in ctrl_sem_parts]
    edge_slices = _build_slices(edge_lengths)
    ctrl_slices = _build_slices(ctrl_lengths)
    part_slices = _build_slices(part_lengths)

    part_offsets = [part_slice.start for part_slice in part_slices]
    z_geo_cat = torch.cat(z_geo_parts, dim=0)
    z_sem_cat = torch.cat(z_sem_parts, dim=0)
    material_cat = torch.cat(material_parts, dim=0)
    part_features_cat = torch.cat(part_feature_parts, dim=0)
    edge_part_idx_cat = torch.cat(
        [batch["edge_part_idx"][i].to(device) + part_offsets[i] for i in range(num_cases)],
        dim=0,
    )

    model_impl = _unwrap_model(model)
    z_mat_parts_cat = model_impl.encode_material_prior(
        material_cat,
        part_features_cat,
        prior_source=prior_source,
    )
    z_mat_cat = z_mat_parts_cat[edge_part_idx_cat]
    z_geo_enc_cat = model_impl.geo_encoder(z_geo_cat)
    raw_log_k_cat = model_impl.decoder(z_geo_enc_cat, z_sem_cat, z_mat_cat)
    log_k_min = float(np.log(1e3))
    log_k_max = float(np.log(1e5))
    log_k_cat = torch.sigmoid(raw_log_k_cat) * (log_k_max - log_k_min) + log_k_min

    ctrl_log_k_cat = None
    if sum(ctrl_lengths) > 0:
        ctrl_sem_cat = torch.cat(ctrl_sem_parts, dim=0)
        ctrl_rest_cat = torch.cat(ctrl_rest_parts, dim=0)
        ctrl_part_idx_cat = torch.cat(
            [batch["ctrl_part_idx"][i].to(device) + part_offsets[i] for i in range(num_cases)],
            dim=0,
        )
        ctrl_z_mat_cat = z_mat_parts_cat[ctrl_part_idx_cat]
        raw_ctrl_log_k_cat = model_impl.ctrl_decoder(ctrl_sem_cat, ctrl_z_mat_cat, ctrl_rest_cat)
        ctrl_log_k_cat = torch.sigmoid(raw_ctrl_log_k_cat) * (log_k_max - log_k_min) + log_k_min

    z_sem_global_cat = torch.cat(z_sem_global_parts, dim=0)

    case_outputs: List[Dict[str, torch.Tensor]] = []
    for idx in range(num_cases):
        global_out = model_impl.global_decoder(
            z_sem_global_cat[idx:idx + 1],
            z_mat_parts_cat[part_slices[idx]],
        )
        case_out = {
            "log_k": log_k_cat[edge_slices[idx]],
            "collide_elas": global_out["collide_elas"],
            "collide_fric": global_out["collide_fric"],
            "collide_object_elas": global_out["collide_object_elas"],
            "collide_object_fric": global_out["collide_object_fric"],
            "collision_dist": global_out["collision_dist"],
            "dashpot_damping": global_out["dashpot_damping"],
            "drag_damping": global_out["drag_damping"],
        }
        if ctrl_log_k_cat is not None and ctrl_lengths[idx] > 0:
            case_out["ctrl_log_k"] = ctrl_log_k_cat[ctrl_slices[idx]]
        case_outputs.append(case_out)
    return case_outputs


def enrich_model_output(
    model_out: Dict[str, torch.Tensor],
    batch,
    idx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Keep a single output path now that all physics parameters are model-predicted."""
    return dict(model_out)




def _clamp_log_stiffness(log_k: torch.Tensor, runtime: CaseRuntime) -> torch.Tensor:
    """Keep log stiffness in a finite range before Warp exponentiates it."""
    max_k = float(max(getattr(runtime.sim, "spring_Y_max", 1e5), 1.0))
    min_k = float(max(getattr(runtime.sim, "spring_Y_min", 0.0), 1e-6))
    min_log = float(np.log(min_k))
    max_log = float(np.log(max_k))
    return torch.nan_to_num(log_k, nan=0.0, posinf=max_log, neginf=min_log).clamp(min_log, max_log)


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def compute_part_kl_loss(
    model: nn.Module,
    batch,
    idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Part-level semantic consistency: semantic part features -> material distribution."""
    model_impl = _unwrap_model(model)
    if not hasattr(model_impl, "predict_part_material_logits"):
        return torch.zeros((), device=device)

    part_features = batch["part_features"][idx].to(device)
    target_material_dist = batch["material_dist"][idx].to(device)
    logits = model_impl.predict_part_material_logits(part_features)
    log_probs = F.log_softmax(logits, dim=-1)
    target = target_material_dist.clamp_min(1e-8)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(log_probs, target, reduction="batchmean")


def compute_prior_distill_weight(epoch: int, args) -> float:
    base = float(getattr(args, "lambda_prior_distill", 0.0))
    if base <= 0:
        base = float(getattr(args, "lambda_part_kl", 0.0))
    start_epoch = int(getattr(args, "prior_distill_start_epoch", 0))
    if start_epoch <= 0:
        start_epoch = int(getattr(args, "part_kl_start_epoch", 0))
    if base <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    return base


def compute_phys_prior_weight(epoch: int, args) -> float:
    if not bool(getattr(args, "use_phys_prior", False)):
        return 0.0
    base = float(getattr(args, "lambda_phys_prior", 0.0))
    if base <= 0:
        return 0.0
    start_epoch = int(getattr(args, "phys_prior_start_epoch", 0))
    if epoch < start_epoch:
        return 0.0
    return base


def compute_part_physics_prior_loss(
    batch,
    idx: int,
    device: torch.device,
    pred_logk_edges: torch.Tensor,
    edge_part_idx: torch.Tensor,
) -> torch.Tensor:
    part_mu = batch["part_phys_prior_mu"][idx].to(device).view(-1, 1)
    part_log_sigma = batch["part_phys_prior_log_sigma"][idx].to(device).view(-1, 1)
    part_conf = batch["part_phys_prior_conf"][idx].to(device).view(-1, 1)
    if part_mu.numel() == 0 or float(part_conf.max().item()) <= 0.0:
        return torch.zeros((), device=device)

    pred_logk_edges = pred_logk_edges.view(-1, 1)
    edge_part_idx = edge_part_idx.to(device).view(-1).long()
    num_parts = int(part_mu.shape[0])
    pred_part = torch.zeros_like(part_mu)
    counts = torch.zeros((num_parts, 1), device=device, dtype=pred_logk_edges.dtype)
    pred_part.index_add_(0, edge_part_idx, pred_logk_edges)
    ones = torch.ones_like(pred_logk_edges)
    counts.index_add_(0, edge_part_idx, ones)
    pred_part = pred_part / counts.clamp_min(1.0)

    sigma = part_log_sigma.exp().clamp_min(1e-4)
    nll = 0.5 * (((pred_part - part_mu) / sigma) ** 2) + part_log_sigma
    return (part_conf * nll).mean()


def compute_global_physics_prior_loss(
    batch,
    idx: int,
    device: torch.device,
    model_out: Dict[str, torch.Tensor],
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
    nll = 0.5 * (((pred - mu) / sigma) ** 2) + log_sigma
    return conf * nll.mean()


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
    """Train for one epoch using batched model prediction and per-case simulation."""
    model.train()

    epoch_loss = 0.0
    epoch_track = 0.0
    epoch_geo = 0.0
    epoch_render = 0.0
    epoch_graphs = 0
    epoch_prior_kl = 0.0
    epoch_phys_prior = 0.0

    epoch_total_steps = 0
    for batch in train_loader:
        for i in range(len(batch["case_name"])):
            train_frame_val = int(batch["train_frame"][i].item())
            epoch_total_steps += max(train_frame_val - 1, 1)

    pbar = tqdm(total=max(epoch_total_steps, 1), desc="training", dynamic_ncols=True)

    for batch in train_loader:
        optimizer.zero_grad()
        case_model_outs = forward_model_batched(
            model,
            model_type,
            batch,
            device,
            prior_source=args.prior_source,
        )
        batch_backward_tensors: List[torch.Tensor] = []
        batch_backward_grads: List[torch.Tensor] = []
        aux_losses: List[torch.Tensor] = []
        batch_case_count = 0

        for i, raw_model_out in enumerate(case_model_outs):
            case_name = batch["case_name"][i]
            case_model_out = enrich_model_output(raw_model_out, batch, i, device)
            pred_logk = case_model_out["log_k"]

            if case_name not in runtimes:
                train_frame_init = int(batch["train_frame"][i].item())
                print(f"[runtime:init] case={case_name} train_frame={train_frame_init}")
                runtimes[case_name] = CaseRuntime(
                    base_path=args.base_path,
                    case_name=case_name,
                    experiments_optimization_dir=args.experiments_optimization_dir,
                    train_frame=train_frame_init,
                    device=args.device,
                    runtime_root=os.path.join("semantic", "runtime", f"rank_{getattr(args, 'rank', 0)}"),
                    topology_cfg={
                        "use_knn_topology": args.use_knn_topology,
                        "object_knn": args.object_knn,
                        "object_radius": args.object_radius,
                        "object_max_neighbours": args.object_max_neighbours,
                        "controller_radius": args.controller_radius,
                        "controller_max_neighbours": args.controller_max_neighbours,
                    },
                    gaussian_root=getattr(args, "gaussian_root", None),
                )

            runtime = runtimes[case_name]
            sim = runtime.sim

            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)
            cfg.acc_weight = 0.0

            num_object_springs = runtime.num_object_springs
            if pred_logk.shape[0] != num_object_springs:
                raise ValueError(
                    f"{case_name}: pred springs mismatch {pred_logk.shape[0]} vs {num_object_springs}"
                )

            pred_logk_sim = _clamp_log_stiffness(pred_logk, runtime)
            ctrl_logk = case_model_out.get("ctrl_log_k", None)
            if ctrl_logk is not None and ctrl_logk.numel() > 0:
                ctrl_logk_sim = _clamp_log_stiffness(ctrl_logk, runtime)
            else:
                num_ctrl = sim.n_springs - num_object_springs
                default_ctrl = torch.full(
                    (num_ctrl,),
                    float(np.log(max(cfg.init_spring_Y, 1e-6))),
                    device=device,
                    dtype=pred_logk_sim.dtype,
                )
                ctrl_logk_sim = _clamp_log_stiffness(default_ctrl, runtime)
            model_logk = torch.cat([pred_logk_sim.view(-1), ctrl_logk_sim.view(-1)], dim=0)

            if model_logk.numel() != sim.n_springs:
                raise ValueError(
                    f"{case_name}: spring size mismatch {model_logk.numel()} vs {sim.n_springs}"
                )

            sim.set_spring_Y(model_logk.detach())
            sim.set_collide(
                case_model_out["collide_elas"].detach().view(-1),
                case_model_out["collide_fric"].detach().view(-1),
            )
            sim.set_collide_object(
                case_model_out["collide_object_elas"].detach().view(-1),
                case_model_out["collide_object_fric"].detach().view(-1),
            )
            sim.collision_dist = float(case_model_out["collision_dist"].detach())
            sim.dashpot_damping = float(case_model_out["dashpot_damping"].detach())
            sim.drag_damping = float(case_model_out["drag_damping"].detach())

            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
            sim.set_acc_count(False)

            train_frame = int(batch["train_frame"][i].item())
            K = batch["cam0_intrinsics"][i].to(device)
            w2c = batch["cam0_w2c"][i].to(device)
            wh = batch["wh"][i].to(device)
            width = int(wh[0].item())
            height = int(wh[1].item())

            grad_accum = torch.zeros_like(model_logk)
            grad_collide_elas = torch.zeros_like(case_model_out["collide_elas"])
            grad_collide_fric = torch.zeros_like(case_model_out["collide_fric"])
            grad_collide_object_elas = torch.zeros_like(case_model_out["collide_object_elas"])
            grad_collide_object_fric = torch.zeros_like(case_model_out["collide_object_fric"])
            graph_geo = 0.0
            graph_track = 0.0
            graph_render = 0.0
            steps = 0
            lambda_render = float(args.lambda_render)

            for frame_idx in range(1, train_frame):
                sim.set_controller_target(frame_idx)
                if sim.object_collision_flag:
                    sim.update_collision_graph()

                with sim.tape:
                    sim.step()
                    sim.calculate_loss()

                track_val = wp.to_torch(sim.track_loss, requires_grad=False).item()
                geo_val = wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                graph_track += track_val
                graph_geo += geo_val

                pred_points = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=True)
                render_val = torch.zeros((), device=device)
                if runtime.gaussians is not None and lambda_render > 0:
                    gt_color = runtime.load_gt_color_cam0(frame_idx, width=width, height=height)
                    if gt_color is not None:
                        view = _make_gs_view(w2c, K, height, width, device)
                        fg_mask = runtime.load_union_mask_cam0(frame_idx, width=width, height=height)
                        render_val = gaussian_render_l1_loss(
                            pred_points, runtime.sim_init_pts, runtime.gaussians,
                            runtime.knn_indices, runtime.knn_weights_vals,
                            runtime.gs_init_xyz, view, gt_color, runtime.gs_background,
                            mask=fg_mask,
                        )
                graph_render += float(render_val.item())

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

                sim.tape.reset()
                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
                steps += 1
                pbar.update(1)

            if steps == 0:
                pbar.update(1)
                continue

            mean_track = graph_track / steps
            mean_geo = graph_geo / steps
            mean_render = graph_render / steps
            if not np.isfinite(mean_track + mean_geo + mean_render):
                print(f"[warn] non-finite loss on case {case_name} at epoch {epoch+1}; skipping case")
                continue

            grad_scale = getattr(args, 'grad_scale', 1e4)
            grad_input = torch.nan_to_num((grad_accum / float(steps)) * grad_scale, nan=0.0, posinf=0.0, neginf=0.0)
            grad_collide_elas = torch.nan_to_num(grad_collide_elas / float(steps), nan=0.0, posinf=0.0, neginf=0.0)
            grad_collide_fric = torch.nan_to_num(grad_collide_fric / float(steps), nan=0.0, posinf=0.0, neginf=0.0)
            grad_collide_object_elas = torch.nan_to_num(grad_collide_object_elas / float(steps), nan=0.0, posinf=0.0, neginf=0.0)
            grad_collide_object_fric = torch.nan_to_num(grad_collide_object_fric / float(steps), nan=0.0, posinf=0.0, neginf=0.0)

            grad_tensors_to_check = [
                grad_input,
                grad_collide_elas,
                grad_collide_fric,
                grad_collide_object_elas,
                grad_collide_object_fric,
            ]
            if not all(_is_finite_tensor(t) for t in grad_tensors_to_check):
                print(f"[warn] skipping case with non-finite simulation gradients: {case_name}")
                continue

            case_aux_loss = torch.zeros((), device=device)

            prior_kl = compute_part_kl_loss(model, batch, i, device)
            prior_distill_weight = compute_prior_distill_weight(epoch, args)
            if prior_distill_weight > 0:
                case_aux_loss = case_aux_loss + (prior_distill_weight * prior_kl)
            epoch_prior_kl += float(prior_kl.detach().item())

            part_phys_prior = compute_part_physics_prior_loss(
                batch,
                i,
                device,
                pred_logk,
                batch["edge_part_idx"][i],
            )
            global_phys_prior = compute_global_physics_prior_loss(batch, i, device, case_model_out)
            phys_prior = part_phys_prior + global_phys_prior
            phys_prior_weight = compute_phys_prior_weight(epoch, args)
            if phys_prior_weight > 0:
                case_aux_loss = case_aux_loss + (phys_prior_weight * phys_prior)
            epoch_phys_prior += float(phys_prior.detach().item())

            if case_aux_loss.requires_grad:
                aux_losses.append(case_aux_loss)

            batch_backward_tensors.append(model_logk)
            batch_backward_grads.append(grad_input)
            for tensor, grad in [
                (case_model_out["collide_elas"], grad_collide_elas),
                (case_model_out["collide_fric"], grad_collide_fric),
                (case_model_out["collide_object_elas"], grad_collide_object_elas),
                (case_model_out["collide_object_fric"], grad_collide_object_fric),
            ]:
                if tensor.requires_grad:
                    batch_backward_tensors.append(tensor)
                    batch_backward_grads.append(grad)

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
            batch_case_count += 1

            pbar.set_postfix(
                total=f"{total:.4f}",
                track=f"{mean_track:.4f}",
                geo=f"{mean_geo:.4f}",
                render=f"{mean_render:.4f}",
            )

        if batch_case_count == 0:
            continue

        if aux_losses:
            aux_loss = torch.stack(aux_losses).mean()
            aux_loss.backward(retain_graph=bool(batch_backward_tensors))

        if batch_backward_tensors:
            batch_backward_grads = [grad / float(batch_case_count) for grad in batch_backward_grads]
            torch.autograd.backward(
                tensors=batch_backward_tensors,
                grad_tensors=batch_backward_grads,
            )

        bad_param_grad = False
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                bad_param_grad = True
                break
        if bad_param_grad:
            print("[warn] skipping optimizer step due to non-finite parameter gradients")
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    pbar.close()

    n = max(epoch_graphs, 1)
    loss_value = epoch_loss / n
    track_value = epoch_track / n
    geo_value = epoch_geo / n
    render_value = epoch_render / n
    prior_kl_value = epoch_prior_kl / n
    phys_prior_value = epoch_phys_prior / n
    if _is_distributed():
        loss_value = _all_reduce_mean(epoch_loss, epoch_graphs, device=device)
        track_value = _all_reduce_mean(epoch_track, epoch_graphs, device=device)
        geo_value = _all_reduce_mean(epoch_geo, epoch_graphs, device=device)
        render_value = _all_reduce_mean(epoch_render, epoch_graphs, device=device)
        prior_kl_value = _all_reduce_mean(epoch_prior_kl, epoch_graphs, device=device)
        phys_prior_value = _all_reduce_mean(epoch_phys_prior, epoch_graphs, device=device)
        num_graphs_tensor = torch.tensor([epoch_graphs], dtype=torch.long, device=device)
        dist.all_reduce(num_graphs_tensor, op=dist.ReduceOp.SUM)
        epoch_graphs = int(num_graphs_tensor.item())
    return {
        "loss": loss_value,
        "track": track_value,
        "geo": geo_value,
        "render": render_value,
        "prior_kl": prior_kl_value,
        "phys_prior": phys_prior_value,
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
    total_prior_kl = 0.0
    total_phys_prior = 0.0

    # Create visualization directory
    vis_dir = None
    render_cmp_dir = None
    if save_dir is not None:
        vis_dir = os.path.join(save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        render_cmp_dir = os.path.join(save_dir, "render_compare", f"epoch_{epoch:04d}")
        os.makedirs(render_cmp_dir, exist_ok=True)

    for batch in test_loader:
        case_model_outs = forward_model_batched(
            model,
            model_type,
            batch,
            device,
            prior_source=args.prior_source,
        )
        for i, raw_model_out in enumerate(case_model_outs):
            case_name = batch["case_name"][i]
            model_out = enrich_model_output(raw_model_out, batch, i, device)
            pred_logk = model_out["log_k"]
            total_prior_kl += float(compute_part_kl_loss(model, batch, i, device).item())
            total_phys_prior += float((
                compute_part_physics_prior_loss(
                    batch, i, device, pred_logk, batch["edge_part_idx"][i],
                ) + compute_global_physics_prior_loss(batch, i, device, model_out)
            ).item())

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
                    runtime_root=os.path.join("semantic", "runtime", f"rank_{getattr(args, 'rank', 0)}"),
                    topology_cfg={
                        "use_knn_topology": args.use_knn_topology,
                        "object_knn": args.object_knn,
                        "object_radius": args.object_radius,
                        "object_max_neighbours": args.object_max_neighbours,
                        "controller_radius": args.controller_radius,
                        "controller_max_neighbours": args.controller_max_neighbours,
                    },
                    gaussian_root=getattr(args, "gaussian_root", None),
                )

            runtime = runtimes[case_name]
            sim = runtime.sim

            cfg.chamfer_weight = float(args.lambda_geo)
            cfg.track_weight = float(args.lambda_track)
            cfg.acc_weight = 0.0

            num_object_springs = runtime.num_object_springs

            if pred_logk.shape[0] != num_object_springs:
                continue

            pred_logk_sim = _clamp_log_stiffness(pred_logk, runtime)
            # Controller springs: predicted if available, else fallback to baseline
            if "ctrl_log_k" in model_out and model_out["ctrl_log_k"].numel() > 0:
                ctrl_logk = _clamp_log_stiffness(model_out["ctrl_log_k"].view(-1), runtime)
            else:
                num_ctrl = sim.n_springs - num_object_springs
                ctrl_logk = torch.full(
                    (num_ctrl,),
                    float(np.log(max(cfg.init_spring_Y, 1e-6))),
                    device=device,
                    dtype=pred_logk_sim.dtype,
                )
                ctrl_logk = _clamp_log_stiffness(ctrl_logk, runtime)

            full_logk = torch.cat([pred_logk_sim.view(-1), ctrl_logk], dim=0)
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
            save_frame_idx = max(1, train_frame // 2)  # middle frame for comparison image
            cmp_rendered = None
            cmp_gt = None

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

                render_val = torch.zeros(())
                if runtime.gaussians is not None:
                    pred_points_eval = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
                    gt_color = runtime.load_gt_color_cam0(frame_idx, width=width, height=height)
                    if gt_color is not None:
                        view = _make_gs_view(w2c, K, height, width, device)
                        fg_mask = runtime.load_union_mask_cam0(frame_idx, width=width, height=height)
                        capture = (render_cmp_dir is not None and frame_idx == save_frame_idx)
                        with torch.no_grad():
                            if capture:
                                render_val, cmp_rendered, cmp_gt = gaussian_render_l1_loss(
                                    pred_points_eval, runtime.sim_init_pts, runtime.gaussians,
                                    runtime.knn_indices, runtime.knn_weights_vals,
                                    runtime.gs_init_xyz, view, gt_color, runtime.gs_background,
                                    mask=fg_mask, return_rendered=True,
                                )
                            else:
                                render_val = gaussian_render_l1_loss(
                                    pred_points_eval, runtime.sim_init_pts, runtime.gaussians,
                                    runtime.knn_indices, runtime.knn_weights_vals,
                                    runtime.gs_init_xyz, view, gt_color, runtime.gs_background,
                                    mask=fg_mask,
                                )
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

                # Save GT vs rendered comparison image
                if render_cmp_dir is not None and cmp_rendered is not None and cmp_gt is not None:
                    def _to_uint8(t):
                        return (t.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
                    gt_img = _to_uint8(cmp_gt)
                    rend_img = _to_uint8(cmp_rendered)
                    diff = np.abs(gt_img.astype(np.float32) - rend_img.astype(np.float32)).astype(np.uint8)
                    sep = np.ones((height, 4, 3), dtype=np.uint8) * 128
                    compare = np.concatenate([gt_img, sep, rend_img, sep, diff], axis=1)
                    cmp_path = os.path.join(render_cmp_dir, f"{case_name}_frame{save_frame_idx:04d}.png")
                    cv2.imwrite(cmp_path, cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

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
        "prior_kl": total_prior_kl / n,
        "phys_prior": total_phys_prior / n,
        "num_graphs": total_graphs,
    }


def create_model_for_type(
    model_type: str,
    dino_dim: int = 1024,
    geo_input_dim: int = 10,
    num_materials: int = 10,
    prior_encoder_type: str = "codebook",
    prior_hidden_dim: int = 128,
) -> nn.Module:
    """Create model based on type."""
    if model_type == "edge_level":
        return EdgeLevelMaterialPhysics(
            sem_dim=dino_dim,
            dino_dim=dino_dim,
            geo_input_dim=geo_input_dim,
            num_materials=num_materials,
            prior_encoder_type=prior_encoder_type,
            prior_hidden_dim=prior_hidden_dim,
        )
    elif model_type == "part_level":
        return PartLevelMaterialPhysics(
            num_materials=num_materials,
            sem_dim=dino_dim,
            prior_encoder_type=prior_encoder_type,
            prior_hidden_dim=prior_hidden_dim,
        )
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
    parser.add_argument("--save_dir", type=str, default="checkpoints/debug")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint in save_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_type",
        type=str,
        default="edge_level",
        choices=["edge_level", "part_level"],
    )
    parser.add_argument("--num_materials", type=int, default=10,
                        help="Number of VLM material categories")
    parser.add_argument(
        "--prior_encoder_type",
        type=str,
        default="codebook",
        choices=["codebook", "direct_mlp"],
        help="How to encode part-level material priors into the physics latent",
    )
    parser.add_argument(
        "--prior_hidden_dim",
        type=int,
        default=128,
        help="Hidden width for the direct prior MLP encoder",
    )
    parser.add_argument(
        "--prior_source",
        type=str,
        default="vlm",
        choices=["vlm", "distilled"],
        help="Source of the part-level prior used by the physics model",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of cases for training (default 0.8 = 8:2 train/test)")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate on test set every N epochs")

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
    parser.add_argument("--controller_radius", type=float, default=0.04)
    parser.add_argument("--controller_max_neighbours", type=int, default=50)

    # Loss weights
    parser.add_argument("--gaussian_root", type=str, default="gaussian_output",
                        help="Root dir for Gaussian models: <gaussian_root>/<case>/**/point_cloud.ply")
    parser.add_argument("--lambda_render", type=float, default=1.0)
    parser.add_argument("--lambda_track", type=float, default=1.0)
    parser.add_argument("--lambda_geo", type=float, default=1.0)

    # Single case training (for testing model capacity)
    parser.add_argument("--case_name", type=str, default='single_lift_zebra',
                        help="Train on a single case only (for testing model capacity)")

    # Gradient scaling (warp simulation gradients are very small)
    parser.add_argument("--grad_scale", type=float, default=1e3,
                        help="Scale factor for simulation gradients (default: 1e4)")
    parser.add_argument(
        "--lambda_prior_distill",
        type=float,
        default=0.0,
        help="Weight for distilling part_features into the VLM material prior",
    )
    parser.add_argument(
        "--prior_distill_start_epoch",
        type=int,
        default=0,
        help="Epoch to start prior distillation",
    )
    parser.add_argument(
        "--lambda_part_kl",
        type=float,
        default=0.0,
        help="Deprecated alias for --lambda_prior_distill",
    )
    parser.add_argument(
        "--part_kl_start_epoch",
        type=int,
        default=0,
        help="Deprecated alias for --prior_distill_start_epoch",
    )
    parser.add_argument(
        "--use_phys_prior",
        action="store_true",
        help="Enable material-distribution-conditioned continuous physics prior on per-part log stiffness",
    )
    parser.add_argument(
        "--lambda_phys_prior",
        type=float,
        default=0.0,
        help="Weight for the continuous physics-prior loss",
    )
    parser.add_argument(
        "--phys_prior_start_epoch",
        type=int,
        default=0,
        help="Epoch to start applying the continuous physics-prior loss",
    )

    args = parser.parse_args()

    rank, local_rank, world_size, distributed = setup_distributed(args)
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.distributed = distributed
    is_main_process = rank == 0

    set_all_seeds(args.seed + rank)
    os.makedirs(args.save_dir, exist_ok=True) if is_main_process else None
    os.makedirs(os.path.join("semantic", "runtime", f"rank_{rank}"), exist_ok=True)
    wp.set_device(args.device)
    cfg.device = args.device
    device = torch.device(args.device)

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Model: {args.model_type}")
        print(f"Prior encoder: {args.prior_encoder_type} | prior source: {args.prior_source}")
        print(f"Loss weights: track={args.lambda_track}, geo={args.lambda_geo}, render={args.lambda_render}")
        if args.case_name:
            print(f"Single case mode: {args.case_name}")
        print(f"{'='*60}\n")

    local_batch_size = args.batch_size
    if distributed and not args.case_name:
        if args.batch_size < world_size:
            raise ValueError(f"Global batch_size ({args.batch_size}) must be >= world_size ({world_size})")
        if args.batch_size % world_size != 0 and is_main_process:
            print(f"[warn] batch_size={args.batch_size} not divisible by world_size={world_size}; using floor division for local batch size")
        local_batch_size = max(1, args.batch_size // world_size)
        if is_main_process:
            print(f"[DDP] world_size={world_size} local_batch_size={local_batch_size} global_batch_size~={local_batch_size * world_size}")

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
        controller_radius=args.controller_radius,
        controller_max_neighbours=args.controller_max_neighbours,
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
        if is_main_process:
            print(f"[Single Case] Training on: {args.case_name}")
    else:
        # Random 8:2 split of cases (train_ratio=0.8, overridable via --train_ratio)
        full_dataset, train_loader, test_loader = create_train_test_dataloaders(
            cfg=dataset_cfg,
            batch_size=local_batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

    # Get dimensions from dataset
    sample = full_dataset[0]
    geo_input_dim = sample["z_geo"].shape[1]
    dino_dim = sample["z_sem"].shape[1]

    if is_main_process:
        print(f"Geometry feature dim: {geo_input_dim}")
        print(f"DINO feature dim: {dino_dim}")

    num_materials = sample["material_dist"].shape[1]
    if is_main_process:
        print(f"Number of material codes: {num_materials}")

    # Create model
    model = create_model_for_type(
        args.model_type,
        dino_dim=dino_dim,
        geo_input_dim=geo_input_dim,
        num_materials=num_materials,
        prior_encoder_type=args.prior_encoder_type,
        prior_hidden_dim=args.prior_hidden_dim,
    )
    model = model.to(device)
    if is_main_process:
        print_model_summary(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None, find_unused_parameters=True)
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Shared runtimes for train and test
    runtimes: Dict[str, CaseRuntime] = {}

    # Pre-warm all CaseRuntimes before training to avoid lazy-init during AllReduce.
    # Each rank builds all runtimes independently (no collective ops here), so no NCCL timeout.
    topology_cfg_prewarm = {
        "use_knn_topology": args.use_knn_topology,
        "object_knn": args.object_knn,
        "object_radius": args.object_radius,
        "object_max_neighbours": args.object_max_neighbours,
        "controller_radius": args.controller_radius,
        "controller_max_neighbours": args.controller_max_neighbours,
    }
    print(f"[prewarm] building runtimes for {len(full_dataset.samples)} cases on rank {rank} ...")
    for s in full_dataset.samples:
        cname = s["case_name"]
        if cname not in runtimes:
            tf = int(s["train_frame"]) if not isinstance(s["train_frame"], int) else s["train_frame"]
            print(f"[prewarm] rank={rank} case={cname}")
            runtimes[cname] = CaseRuntime(
                base_path=args.base_path,
                case_name=cname,
                experiments_optimization_dir=args.experiments_optimization_dir,
                train_frame=tf,
                device=args.device,
                runtime_root=os.path.join("semantic", "runtime", f"rank_{rank}"),
                topology_cfg=topology_cfg_prewarm,
                gaussian_root=getattr(args, "gaussian_root", None),
            )
    print(f"[prewarm] rank={rank} done, {len(runtimes)} runtimes built.")

    best_test_loss = float('inf')
    best_epoch = 0
    train_history = []
    test_history = []
    start_epoch = 0

    if args.resume:
        resume_path = _find_latest_checkpoint(args.save_dir)
        if resume_path is None:
            raise FileNotFoundError(
                f"--resume was set, but no checkpoint_epoch_*.pth was found in {args.save_dir}"
            )

        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        start_epoch = int(checkpoint.get("epoch", 0))
        best_test_loss = checkpoint.get("best_test_loss", best_test_loss)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        train_history = checkpoint.get("train_history", train_history)
        test_history = checkpoint.get("test_history", test_history)
        print(f"Resumed from checkpoint: {resume_path} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
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

        if is_main_process:
            print(
                f"[Epoch {epoch+1:04d}] Train: "
                f"loss={train_metrics['loss']:.6f} "
                f"track={train_metrics['track']:.6f} "
                f"geo={train_metrics['geo']:.6f} "
                f"render={train_metrics['render']:.6f} "
                f"prior_kl={train_metrics['prior_kl']:.6f}"
            )

        # Evaluate on test set periodically
        if epoch % args.eval_every == 0 or (epoch + 1) == args.epochs:
            if is_main_process:
                test_metrics = evaluate_physics(
                    model=_unwrap_model(model),
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
                    f"render={test_metrics['render']:.6f} "
                    f"prior_kl={test_metrics['prior_kl']:.6f}"
                )

                if test_metrics["total"] < best_test_loss:
                    best_test_loss = test_metrics["total"]
                    best_epoch = epoch + 1
            if distributed:
                dist.barrier()

        # Save checkpoint every 50 epochs
        if is_main_process and (epoch + 1) % args.eval_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_type": args.model_type,
                "model_state_dict": _unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "geo_input_dim": geo_input_dim,
                "dino_dim": dino_dim,
                "num_materials": num_materials,
                "train_metrics": train_metrics,
                "best_test_loss": best_test_loss,
                "best_epoch": best_epoch,
                "train_history": train_history,
                "test_history": test_history,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

    # Final evaluation
    final_metrics = None
    if is_main_process:
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set")
        print(f"{'='*60}")

        final_metrics = evaluate_physics(
            model=_unwrap_model(model),
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

    if distributed:
        dist.barrier()

    # Save final checkpoint only
    if is_main_process:
        ckpt = {
        "epoch": args.epochs,
        "model_type": args.model_type,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "geo_input_dim": geo_input_dim,
        "dino_dim": dino_dim,
        "num_materials": num_materials,
        "final_metrics": final_metrics,
        "best_test_loss": best_test_loss,
        "best_epoch": best_epoch,
        "train_history": train_history,
        "test_history": test_history,
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

    cleanup_distributed()

if __name__ == "__main__":
    main()
# python semantic/train_models.py 
# --save_dir checkpoints/parallel_modes_20260414/part_level_double_lift_cloth_1_kl001_lr3e4_topomatch_physprior 
# --model_type part_level 
# --case_name double_lift_cloth_1
# --batch_size 4 --num_workers 0 
# --epochs 80 --eval_every 10 
# --device cuda:0 
# --lambda_render 1.0 --lambda_track 1.0 --lambda_geo 1.0 
# --grad_scale 1000.0
