import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger


def _load_original_playground_cfg(
    base_path: str, case_name: str, experiments_optimization_dir: str
) -> None:
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    # Keep the legacy spring topology so explicit physics checkpoints from
    # experiments/results match the simulator spring count.
    cfg.use_edge_gating = False
    cfg.use_knn_topology = False
    cfg.sem_cache_dir = "__disabled__"

    use_edge_gating = getattr(cfg, "use_edge_gating", False)
    if not use_edge_gating:
        optimal_path = os.path.join(
            experiments_optimization_dir, case_name, "optimal_params.pkl"
        )
        logger.info(f"Load optimal parameters from: {optimal_path}")
        assert os.path.exists(
            optimal_path
        ), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    with open(os.path.join(base_path, case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])

    with open(os.path.join(base_path, case_name, "metadata.json"), "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive_playground with explicit physics and Gaussian checkpoints"
    )
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--base_path", default="data/different_types")
    parser.add_argument("--physics_path", required=True)
    parser.add_argument("--gs_path", required=True)
    parser.add_argument(
        "--experiments_optimization_dir", default="experiments_optimization"
    )
    parser.add_argument("--runtime_dir", default="temp_experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument("--inv_ctrl", action="store_true")
    parser.add_argument("--virtual_key_input", action="store_true")
    parser.add_argument(
        "--virtual_screen",
        action="store_true",
        help="Alias of --virtual_key_input for remote/virtual keyboard control",
    )
    parser.add_argument("--bg_img_path", default="data/bg.png")
    args = parser.parse_args()

    _load_original_playground_cfg(
        args.base_path, args.case_name, args.experiments_optimization_dir
    )
    cfg.explicit_topology_path = None
    physics_ckpt = torch.load(args.physics_path, map_location="cpu", weights_only=True)
    if "edges" in physics_ckpt:
        cfg.explicit_topology_path = args.physics_path

    case_dir = os.path.join(args.base_path, args.case_name)
    cfg.bg_img_path = args.bg_img_path

    base_dir = os.path.join(args.runtime_dir, args.case_name)
    logger.set_log_file(path=base_dir, name="inference_log")

    trainer = InvPhyTrainerWarp(
        data_path=os.path.join(case_dir, "final_data.pkl"),
        base_dir=base_dir,
        pure_inference_mode=True,
        device=args.device,
    )

    trainer.interactive_playground(
        model_path=args.physics_path,
        gs_path=args.gs_path,
        n_ctrl_parts=args.n_ctrl_parts,
        inv_ctrl=args.inv_ctrl,
        virtual_key_input=(args.virtual_key_input or args.virtual_screen),
    )


if __name__ == "__main__":
    main()
