import argparse
import json
import os
import pickle
import sys
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger


def _load_pose_json(path: str):
    with open(path, "r") as f:
        pose = json.load(f)
    if "pose" in pose:
        pose = pose["pose"][0]
    return pose


def _infer_export_camera_paths(data_path: str):
    data_path = os.path.abspath(data_path)
    if not data_path.endswith("_final_data.pkl"):
        return None, None
    prefix = data_path[: -len("_final_data.pkl")]
    metadata_path = prefix + "_metadata.json"
    calibrate_path = prefix + "_calibrate.pkl"
    if os.path.isfile(metadata_path) and os.path.isfile(calibrate_path):
        return metadata_path, calibrate_path
    return None, None


def _configure_runtime(
    data_path: str,
    physics_path: str,
    width: int,
    height: int,
    pose_json: Optional[str],
    bg_img_path: str,
    max_render_dim: int,
):
    cfg.load_from_yaml("configs/real.yaml")
    cfg.use_edge_gating = False
    cfg.use_knn_topology = False
    cfg.sem_cache_dir = "__disabled__"
    cfg.explicit_topology_path = None

    physics_ckpt = torch.load(physics_path, map_location="cpu", weights_only=True)
    if "edges" in physics_ckpt:
        cfg.explicit_topology_path = physics_path

    intrinsic = np.array(
        [[width, 0.0, width / 2.0], [0.0, height, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    w2c = np.eye(4, dtype=np.float32)
    metadata_path, calibrate_path = _infer_export_camera_paths(data_path)
    if pose_json is not None and os.path.isfile(pose_json):
        pose = _load_pose_json(pose_json)
        intrinsic = np.array(pose["intrinsic"], dtype=np.float32)
        intrinsic[0, :] *= float(width)
        intrinsic[1, :] *= float(height)
        w2c = np.array(pose["extrinsic"], dtype=np.float32)
    elif metadata_path is not None and calibrate_path is not None:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        with open(calibrate_path, "rb") as f:
            c2ws = pickle.load(f)
        width, height = map(int, metadata["WH"])
        intrinsic = np.array(metadata["intrinsics"][0], dtype=np.float32)
        w2c = np.linalg.inv(np.array(c2ws[0], dtype=np.float32))

    longest_dim = max(width, height)
    if max_render_dim > 0 and longest_dim > max_render_dim:
        scale = float(max_render_dim) / float(longest_dim)
        width = max(int(round(width * scale)), 1)
        height = max(int(round(height * scale)), 1)
        intrinsic[0, :] *= scale
        intrinsic[1, :] *= scale

    cfg.WH = [width, height]
    cfg.intrinsics = np.array([intrinsic], dtype=np.float32)
    cfg.w2cs = np.array([w2c], dtype=np.float32)
    cfg.c2ws = np.array([np.linalg.inv(w2c)], dtype=np.float32)
    cfg.bg_img_path = bg_img_path


def main():
    parser = argparse.ArgumentParser(
        description="Launch object-point drag interaction directly from a minimal data package and explicit physics checkpoint."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--physics_path", required=True)
    parser.add_argument("--gs_path", required=True)
    parser.add_argument("--pose_json", default=None)
    parser.add_argument("--runtime_dir", default="temp_experiments/direct_object_drag")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pick_radius_px", type=float, default=30.0)
    parser.add_argument("--bg_img_path", default="data/bg.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max_render_dim", type=int, default=1280)
    parser.add_argument("--method", choices=["controller", "point"], default="controller")
    args = parser.parse_args()

    _configure_runtime(
        data_path=args.data_path,
        physics_path=args.physics_path,
        width=args.width,
        height=args.height,
        pose_json=args.pose_json,
        bg_img_path=args.bg_img_path,
        max_render_dim=args.max_render_dim,
    )

    os.makedirs(args.runtime_dir, exist_ok=True)
    logger.set_log_file(path=args.runtime_dir, name="inference_log")

    trainer = InvPhyTrainerWarp(
        data_path=args.data_path,
        base_dir=args.runtime_dir,
        pure_inference_mode=True,
        device=args.device,
    )
    trainer.object_point_drag_playground(
        model_path=args.physics_path,
        gs_path=args.gs_path,
        pick_radius_px=args.pick_radius_px,
        pose_json_path=args.pose_json,
        method=args.method,
    )


if __name__ == "__main__":
    main()
