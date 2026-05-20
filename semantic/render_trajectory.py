"""Render a precomputed trajectory pkl as a visualization video.

Useful for visualizing first-order per-scene optimization output and
comparing with our model's predictions.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_models import CaseRuntime
from qqtt.utils import cfg, visualize_pc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_pkl", required=True, help="trajectory pkl: ndarray [F, V, 3]")
    p.add_argument("--case_name", required=True)
    p.add_argument("--base_path", default="data/different_types")
    p.add_argument("--experiments_optimization_dir", default="experiments_optimization")
    p.add_argument("--out_video", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train_frame", type=int, default=0,
                   help="(for runtime init only; rendering uses full trajectory)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.traj_pkl, "rb") as f:
        traj = pickle.load(f)
    if not isinstance(traj, np.ndarray):
        raise ValueError(f"Expected ndarray trajectory, got {type(traj)}")
    print(f"[traj] {args.traj_pkl}  shape={traj.shape}  dtype={traj.dtype}")

    if args.train_frame <= 0:
        args.train_frame = max(int(traj.shape[0]) - 1, 1)

    topo = {
        "use_knn_topology": False,
        "object_knn": 30, "object_radius": 0.02, "object_max_neighbours": 30,
        "controller_radius": 0.04, "controller_max_neighbours": 50,
    }
    runtime = CaseRuntime(
        base_path=args.base_path,
        case_name=args.case_name,
        experiments_optimization_dir=args.experiments_optimization_dir,
        train_frame=args.train_frame,
        device=args.device,
        runtime_root=os.path.join("semantic", "runtime", "render_traj"),
        topology_cfg=topo,
        gaussian_root=None,
    )

    num_all_points = runtime.trainer.num_all_points
    F = int(traj.shape[0])
    if traj.shape[1] < num_all_points:
        raise ValueError(
            f"trajectory has {traj.shape[1]} verts but runtime expects "
            f"{num_all_points} (object+controller); dataset/topology mismatch?"
        )

    K = cfg.intrinsics[0]
    w2c = cfg.w2cs[0]
    wh = cfg.WH
    if isinstance(wh, list) and len(wh) == 2 and not hasattr(wh[0], "__len__"):
        width, height = int(wh[0]), int(wh[1])
    else:
        width, height = int(wh[0][0]), int(wh[0][1])

    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    visualize_pc(
        torch.from_numpy(traj[:, :num_all_points, :]),
        runtime.trainer.object_colors,
        runtime.trainer.controller_points,
        visualize=False,
        save_video=True,
        save_path=args.out_video,
        width=width, height=height,
        intrinsic=K, w2c=w2c,
        overlay_path=os.path.join(args.base_path, args.case_name, "color"),
    )
    print(f"[vis] saved {args.out_video}  ({F} frames)")


if __name__ == "__main__":
    main()
