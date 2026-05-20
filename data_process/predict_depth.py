"""Per-frame metric depth via Depth Anything V2, scale-calibrated against
the Cupid mesh rendered at frame 0.

Inputs:
    <base_path>/<case>/color/<cam_idx>/<frame>.png
    <results_dir>/<case>/cupid/mesh0.glb
    <results_dir>/<case>/cupid/pose.json
    <base_path>/<case>/mask/mask_info_<cam_idx>.json
    <base_path>/<case>/mask/<cam_idx>/<obj_idx>/0.png    (object mask for calib)

Outputs:
    <base_path>/<case>/depth/<cam_idx>/<frame>.npy        (float32, millimeters)
    <base_path>/<case>/depth/calib.json                    (fit details)

Calibration:
    The Cupid mesh gives us a sparse "metric depth" anchor at frame 0
    (vertices projected through Cupid pose). Depth Anything V2 outputs a
    relative inverse-depth-like value. We fit a single linear scale+shift
    on frame 0 in inverse-depth space:
        1 / z_metric  ≈  a * da  +  b
    The fit is done over pixels that are (object-mask) AND (have a Cupid
    mesh hit). Then we apply (a, b) to every frame's DA output to obtain
    per-frame metric depth.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import pipeline

CONTROLLER_NAME = "hand"


def render_mesh_depth(mesh_path: str, pose: dict, W: int, H: int) -> np.ndarray:
    """Project mesh vertices through Cupid camera, z-buffer into a sparse depth image.

    Cupid's intrinsics are normalized to the pad-to-square space; we undo
    that to land in original-image pixel coordinates.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    V = np.asarray(mesh.vertices, dtype=np.float64)
    ext = np.asarray(pose["extrinsic"], dtype=np.float64)
    intr = np.asarray(pose["intrinsic"], dtype=np.float64)
    S = max(W, H)
    v_cam = (ext[:3, :3] @ V.T).T + ext[:3, 3]
    u = intr[0, 0] * v_cam[:, 0] / v_cam[:, 2] + intr[0, 2]
    v = intr[1, 1] * v_cam[:, 1] / v_cam[:, 2] + intr[1, 2]
    px = u * S - (S - W) / 2.0
    py = v * S - (S - H) / 2.0
    ok = (v_cam[:, 2] > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    depth = np.full((H, W), np.nan, dtype=np.float32)
    pxi = px[ok].astype(int)
    pyi = py[ok].astype(int)
    z = v_cam[ok, 2].astype(np.float32)
    # Z-buffer (keep nearest)
    for i in range(len(z)):
        cur = depth[pyi[i], pxi[i]]
        if np.isnan(cur) or z[i] < cur:
            depth[pyi[i], pxi[i]] = z[i]
    return depth


def fit_inv_scale(da: np.ndarray, ref_depth: np.ndarray, mask: np.ndarray):
    """Fit  1/z_metric ≈ a*da + b  over (mask & ref_depth_valid) pixels.

    Returns (a, b, residual_median_m, n_anchors).
    """
    valid = mask & ~np.isnan(ref_depth)
    n = int(valid.sum())
    if n < 100:
        raise RuntimeError(
            f"Too few calibration anchors: {n} pixels (need >=100). "
            "Check object mask vs Cupid mesh overlap."
        )
    x = da[valid].astype(np.float64)
    y_inv = 1.0 / ref_depth[valid].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y_inv, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    # residual in METERS (after inversion)
    inv_pred = a * x + b
    z_pred = 1.0 / np.maximum(inv_pred, 1e-3)
    residual = np.abs(z_pred - ref_depth[valid].astype(np.float64))
    return a, b, float(np.median(residual)), n


def resize_to(da: np.ndarray, H: int, W: int) -> np.ndarray:
    if da.shape == (H, W):
        return da
    t = torch.from_numpy(da)[None, None].float()
    out = torch.nn.functional.interpolate(
        t, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", required=True)
    p.add_argument("--case_name", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--cam_idx", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HF model id for Depth Anything V2.",
    )
    p.add_argument(
        "--controller_name", default=CONTROLLER_NAME,
        help="mask_info entries with this label are skipped when picking object mask.",
    )
    args = p.parse_args()

    case_dir = os.path.join(args.base_path, args.case_name)
    color_dir = os.path.join(case_dir, "color", str(args.cam_idx))
    depth_dir = os.path.join(case_dir, "depth", str(args.cam_idx))
    os.makedirs(depth_dir, exist_ok=True)

    # Locate object mask for calibration region (largest non-controller).
    mask_info_path = os.path.join(
        case_dir, "mask", f"mask_info_{args.cam_idx}.json"
    )
    if not os.path.isfile(mask_info_path):
        raise FileNotFoundError(mask_info_path)
    mi = json.load(open(mask_info_path))
    obj_idx = None
    for k, name in mi.items():
        if name == args.controller_name:
            continue
        obj_idx = int(k)
        break
    if obj_idx is None:
        raise RuntimeError(f"no object mask found in {mask_info_path}")
    obj_mask_path = os.path.join(
        case_dir, "mask", str(args.cam_idx), str(obj_idx), "0.png"
    )
    obj_mask0 = np.array(Image.open(obj_mask_path).convert("L")) > 127
    print(f"[predict_depth] object mask: {obj_mask0.sum()} px (from {obj_mask_path})")

    # Cupid pose + mesh
    cupid_dir = os.path.join(args.results_dir, args.case_name, "cupid")
    pose = json.load(open(os.path.join(cupid_dir, "pose.json")))
    mesh_path = os.path.join(cupid_dir, "mesh0.glb")
    if not os.path.isfile(mesh_path):
        # Allow alternative names
        candidates = sorted(glob.glob(os.path.join(cupid_dir, "mesh*.glb")))
        if not candidates:
            raise FileNotFoundError(f"No Cupid mesh under {cupid_dir}")
        mesh_path = candidates[0]
    print(f"[predict_depth] cupid mesh: {mesh_path}")

    # Frame list
    pngs = sorted(
        glob.glob(os.path.join(color_dir, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if not pngs:
        raise FileNotFoundError(f"No frames in {color_dir}")
    W, H = Image.open(pngs[0]).size
    print(f"[predict_depth] {len(pngs)} frames at {W}x{H}")

    # Load Depth Anything
    print(f"[predict_depth] loading {args.model} on {args.device}")
    pipe = pipeline("depth-estimation", model=args.model, device=args.device)

    # ---------- 1) Cupid mesh depth (frame 0 anchor) ----------
    print("[predict_depth] rendering Cupid mesh depth at frame 0 ...")
    ref_depth = render_mesh_depth(mesh_path, pose, W, H)
    print(
        f"  anchor pixels (mesh hit): {(~np.isnan(ref_depth)).sum()}, "
        f"object-overlap: {((obj_mask0) & ~np.isnan(ref_depth)).sum()}"
    )

    # ---------- 2) DA frame 0 -> fit scale+shift ----------
    img0 = Image.open(pngs[0]).convert("RGB")
    da0 = np.asarray(pipe(img0)["predicted_depth"], dtype=np.float32)
    da0 = resize_to(da0, H, W)
    a, b, res, n = fit_inv_scale(da0, ref_depth, obj_mask0)
    print(
        f"[predict_depth] fit:  1/z_metric ≈ {a:.6e}*da + {b:.6e}   "
        f"(n_anchors={n}, residual median={res * 100:.2f} cm)"
    )

    calib = {
        "model": args.model,
        "image_wh": [W, H],
        "fit_a": a,
        "fit_b": b,
        "fit_n_anchors": int(n),
        "fit_residual_median_m": res,
    }
    with open(os.path.join(os.path.dirname(depth_dir), "calib.json"), "w") as f:
        json.dump(calib, f, indent=2)

    # ---------- 3) Predict + save every frame ----------
    for k, pngp in enumerate(pngs):
        idx = int(os.path.splitext(os.path.basename(pngp))[0])
        img = Image.open(pngp).convert("RGB")
        da = np.asarray(pipe(img)["predicted_depth"], dtype=np.float32)
        da = resize_to(da, H, W)
        inv = a * da + b
        inv = np.maximum(inv, 1e-3)
        z_m = 1.0 / inv  # meters
        # Save in millimeters as float32 (consumers divide by 1000 to get m).
        np.save(os.path.join(depth_dir, f"{idx}.npy"),
                (z_m * 1000.0).astype(np.float32))
        if k % 10 == 0 or k == len(pngs) - 1:
            med = float(np.median(z_m[obj_mask0])) if obj_mask0.any() else float("nan")
            print(f"  frame {idx:4d}  median object z = {med:.3f} m")

    print(f"[predict_depth] DONE -> {depth_dir}")


if __name__ == "__main__":
    main()
