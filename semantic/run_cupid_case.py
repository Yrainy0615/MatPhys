"""
Run Cupid 3D reconstruction for a single case.

Usage (from project root, inside cupid conda env):
    python semantic/run_cupid_case.py \
        --case_name single_lift_cloth \
        --base_path data/different_types \
        --output_dir results

Outputs saved to: results/[case_name]/cupid/
    - gaussians.pt   : dict with xyz, scale, rotation, opacity, color tensors
    - pose.json       : camera extrinsic (4x4) and intrinsic (3x3)
    - render_input_view.png : side-by-side input vs re-rendered
    - mesh.glb        : exported mesh
"""

import argparse
import glob
import json
import os
import sys

import imageio
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

os.environ["SPCONV_ALGO"] = "native"

CUPID_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "Cupid")
CUPID_ROOT = os.path.abspath(CUPID_ROOT)
if CUPID_ROOT not in sys.path:
    sys.path.insert(0, CUPID_ROOT)

from cupid.pipelines import Cupid3DPipeline
from cupid.utils import render_utils, sample_utils
from cupid.utils.align_utils import save_mesh


CONTROLLER_NAME = "hand"


def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def save_gaussian_dict(gs, path):
    """Extract core Gaussian attributes and save as a plain tensor dict."""
    data = {
        "xyz": gs.get_xyz.detach().cpu(),
        "scale": gs.get_scaling.detach().cpu(),
        "rotation": gs.get_rotation.detach().cpu(),
        "opacity": gs.get_opacity.detach().cpu(),
        "color": gs._features_dc.detach().cpu(),
        "aabb": gs.aabb.detach().cpu(),
        "init_params": gs.init_params,
    }
    torch.save(data, path)
    print(f"  [saved] gaussians -> {path}  ({data['xyz'].shape[0]} points)")


def save_gaussian_ply(gs, path):
    """Save Gaussian representation as a renderable PLY file."""
    gs.save_ply(path)
    print(f"  [saved] gaussian ply -> {path}")


def save_pose_json(pose, path):
    """Save camera pose (extrinsic + intrinsic + model_scale) as JSON."""
    data = make_serializable(pose)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved] pose -> {path}")


def load_object_masked_image(case_dir):
    """Load camera-0 frame and apply the non-hand object mask."""
    frame_candidates = sorted(
        glob.glob(os.path.join(case_dir, "color", "0", "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if not frame_candidates:
        raise FileNotFoundError(
            f"no input frames found under: {os.path.join(case_dir, 'color', '0')}"
        )
    input_img_path = frame_candidates[0]
    frame_stem = os.path.splitext(os.path.basename(input_img_path))[0]
    mask_info_path = os.path.join(case_dir, "mask", "mask_info_0.json")

    image = Image.open(input_img_path).convert("RGB")

    if not os.path.exists(mask_info_path):
        return sample_utils.pad_to_square(image), None

    with open(mask_info_path, "r") as f:
        mask_info = json.load(f)

    object_idx = None
    for key, value in mask_info.items():
        if value != CONTROLLER_NAME:
            if object_idx is not None:
                raise ValueError(f"{case_dir}: more than one non-hand object detected")
            object_idx = int(key)

    if object_idx is None:
        return sample_utils.pad_to_square(image), None

    mask_path = os.path.join(case_dir, "mask", "0", str(object_idx), f"{frame_stem}.png")
    if not os.path.exists(mask_path):
        return sample_utils.pad_to_square(image), None

    mask = Image.open(mask_path)
    if mask.mode != "L":
        mask = mask.split()[-1]
    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(mask)
    return sample_utils.pad_to_square(image_rgba), mask_path


def run_case(pipeline, case_name, base_path, output_dir):
    case_dir = os.path.join(base_path, case_name)
    if not os.path.exists(case_dir):
        print(f"  [skip] case dir not found: {case_dir}")
        return

    out_dir = os.path.join(output_dir, case_name, "cupid")
    os.makedirs(out_dir, exist_ok=True)

    try:
        image, mask_path = load_object_masked_image(case_dir)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return
    if mask_path is not None:
        print(f"  [input] using object-masked image: {mask_path}")
    else:
        print("  [input] object mask not found, falling back to raw image")

    print(f"  [run] Cupid pipeline ...")
    outputs = pipeline.run(image)

    gs = outputs["gaussian"][0]
    pose = outputs["pose"][0]

    save_gaussian_dict(gs, os.path.join(out_dir, "gaussians.pt"))
    save_gaussian_ply(gs, os.path.join(out_dir, "gaussians.ply"))
    save_pose_json(pose, os.path.join(out_dir, "pose.json"))

    render_rgb = render_utils.render_pose(gs, pose)["color"][0]
    input_rgba = image.convert("RGBA")
    input_rgb = Image.alpha_composite(
        Image.new("RGBA", input_rgba.size, (0, 0, 0, 255)), input_rgba
    )
    input_rgb = np.array(
        input_rgb.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
    )
    side_by_side = np.concatenate([input_rgb, render_rgb], axis=1)
    imageio.imwrite(os.path.join(out_dir, "render_input_view.png"), side_by_side)

    save_mesh(
        all_outputs=outputs,
        poses=outputs.pop("pose"),
        output_dir=out_dir,
    )

    input_copy_path = os.path.join(out_dir, "input_masked.png")
    image.save(input_copy_path)

    print(f"  [done] {case_name}")


def main():
    parser = argparse.ArgumentParser(description="Run Cupid for one or all cases")
    parser.add_argument("--case_name", type=str, default=None,
                        help="Single case name. If omitted, run all cases under base_path.")
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    print("[init] Loading Cupid pipeline ...")
    try:
        cupid_model_path = snapshot_download("hbb1/Cupid", local_files_only=True)
        print(f"[init] Using local Cupid snapshot: {cupid_model_path}")
    except Exception:
        cupid_model_path = snapshot_download("hbb1/Cupid")
        print(f"[init] Downloaded Cupid snapshot: {cupid_model_path}")
    pipeline = Cupid3DPipeline.from_pretrained(cupid_model_path)
    pipeline.cuda()

    if args.case_name:
        case_names = [args.case_name]
    else:
        case_names = sorted([
            d for d in os.listdir(args.base_path)
            if os.path.isdir(os.path.join(args.base_path, d))
        ])

    print(f"[info] {len(case_names)} case(s) to process")
    for idx, case_name in enumerate(case_names):
        print(f"\n[{idx+1}/{len(case_names)}] case={case_name}")
        run_case(pipeline, case_name, args.base_path, args.output_dir)

    print("\n[all done]")


if __name__ == "__main__":
    main()
