# Experimental feature to approximate the materials in the spring-mass model.
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    cfg.load_from_yaml("configs/real.yaml")

    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./experiments",
    )
    parser.add_argument("--case_name", type=str, default="double_stretch_sloth")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the best model checkpoint. If not provided, searches in experiments/{case_name}/train/")
    parser.add_argument("--use_knn_topology", type=str, default=None, help="Override use_knn_topology config (true/false)")
    parser.add_argument("--object_knn", type=int, default=None, help="Override object_knn config")
    parser.add_argument("--use_edge_gating", type=str, default=None, help="Override use_edge_gating config (true/false)")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"{args.base_dir}/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    # Auto-detect config from checkpoint before building the simulator
    if args.model_path:
        best_model_path = args.model_path
    else:
        candidates = glob.glob(f"experiments/{case_name}/train/best_*.pth")
        assert candidates, f"No best model found in experiments/{case_name}/train/"
        best_model_path = candidates[0]

    # Detect whether the checkpoint was trained with edge gating
    checkpoint_keys = torch.load(best_model_path, map_location="cpu").keys()
    has_edge_gate = "edge_gate" in checkpoint_keys and "log_a" in checkpoint_keys

    # Override config to match the checkpoint
    if args.use_knn_topology is not None:
        cfg.use_knn_topology = args.use_knn_topology.lower() == "true"
    if args.object_knn is not None:
        cfg.object_knn = args.object_knn
    if args.use_edge_gating is not None:
        cfg.use_edge_gating = args.use_edge_gating.lower() == "true"
    elif not has_edge_gate:
        # Auto-disable edge gating if checkpoint doesn't have it
        logger.info("Checkpoint has no edge_gate/log_a keys; disabling use_edge_gating")
        cfg.use_edge_gating = False
        if args.use_knn_topology is None:
            # If checkpoint was trained without edge gating, it likely used radius-based topology
            logger.info("Disabling use_knn_topology to match old checkpoint")
            cfg.use_knn_topology = False
        # Also disable DINO cluster mask loading since old models didn't use it
        logger.info("Disabling DINO cluster mask to match old checkpoint")
        cfg.sem_cache_dir = "__disabled__"

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    trainer.visualize_material(best_model_path, gaussians_path)
