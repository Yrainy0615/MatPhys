"""
Lift DINOv2 features from the input image to visible Gaussians.

Usage (from project root):
    python semantic/lift_dino_to_gaussian.py \
        --case_name single_lift_cloth_4 \
        --output_dir results

Input:
    results/[case_name]/cupid/gaussians.pt
    results/[case_name]/cupid/pose.json
    results/[case_name]/cupid/input_masked.png

Output:
    results/[case_name]/gaussian_vis/gaussians_vis.pt
    results/[case_name]/gaussian_vis/visible_mask.pt
    results/[case_name]/gaussian_vis/dino_feat_vis.pt
    results/[case_name]/gaussian_vis/feature_render_pca.png
"""

import argparse
import json
import os

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from torchvision import transforms


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x):
    return torch.log(x.clamp(1e-7, 1 - 1e-7) / (1 - x.clamp(1e-7, 1 - 1e-7)))


def save_gaussians_ply(gs, path, color_override=None):
    """
    Save Gaussian attributes as a 3DGS-compatible PLY.

    ``gs`` is the dict from gaussians.pt (activated values).
    ``color_override`` – optional [N, 3] float tensor in [0, 1] to replace SH DC.
    """
    xyz = gs["xyz"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    if color_override is not None:
        f_dc = color_override.detach().cpu().numpy()
    else:
        f_dc = (
            gs["color"]
            .detach()
            .squeeze(1)                # [N, 1, 3] -> [N, 3]
            .cpu()
            .numpy()
        )

    opacities = _inverse_sigmoid(gs["opacity"].detach().cpu()).numpy()
    scales = torch.log(gs["scale"].detach().cpu().clamp_min(1e-8)).numpy()
    rotations = gs["rotation"].detach().cpu().numpy()

    attr_names = ["x", "y", "z", "nx", "ny", "nz",
                  "f_dc_0", "f_dc_1", "f_dc_2",
                  "opacity",
                  "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3"]
    dtype_full = [(a, "f4") for a in attr_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, np.concatenate(
        [xyz, normals, f_dc, opacities, scales, rotations], axis=1
    )))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_gaussians(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pose(path):
    with open(path, "r") as f:
        pose = json.load(f)
    extrinsic = torch.tensor(pose["extrinsic"], dtype=torch.float32)
    intrinsic = torch.tensor(pose["intrinsic"], dtype=torch.float32)
    return extrinsic, intrinsic


def load_image_rgb(path):
    """Load image and composite RGBA over black background to get RGB."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_gaussians(xyz, extrinsic, intrinsic):
    """
    Project 3D gaussian centres to 2D NDC coordinates.

    Cupid stores intrinsics in NDC space where the full image spans [0, 1].
    Projection: u_ndc = fx * X/Z + cx,  v_ndc = fy * Y/Z + cy.

    Returns:
        uv_ndc  – [N, 2] in approximately [0, 1]
        z       – [N]    camera-space depth
    """
    N = xyz.shape[0]
    ones = torch.ones((N, 1), dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)                # [N, 4]
    cam = (extrinsic @ xyz_h.T).T[:, :3]                 # [N, 3]
    z = cam[:, 2].clone()
    uv_h = (intrinsic @ cam.T).T                         # [N, 3]
    uv_ndc = uv_h[:, :2] / uv_h[:, 2:3].clamp_min(1e-8)
    return uv_ndc, z


def compute_visibility_mask(uv_ndc, z, opacity, margin=0.02):
    """In-frustum + positive depth + sufficient opacity."""
    return (
        (uv_ndc[:, 0] >= -margin)
        & (uv_ndc[:, 0] <= 1.0 + margin)
        & (uv_ndc[:, 1] >= -margin)
        & (uv_ndc[:, 1] <= 1.0 + margin)
        & (z > 0)
        & (opacity.squeeze(-1) > 0.01)
    )


# ---------------------------------------------------------------------------
# DINO feature extraction
# ---------------------------------------------------------------------------

def extract_dino_features(image, model_name="dinov2_vitl14_reg",
                          image_size=518, device="cuda"):
    """
    Extract dense DINOv2 patch-token features.

    Returns:
        feat_map  – [1, D, n_patch, n_patch]
        n_patch   – int, patches per side
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)

    xform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_t = xform(image).unsqueeze(0).to(device)
    n_patch = image_size // 14

    with torch.no_grad():
        feats = model(img_t, is_training=True)
        patch_tokens = (
            feats["x_prenorm"][:, model.num_register_tokens + 1:]
            .permute(0, 2, 1)
            .reshape(1, -1, n_patch, n_patch)
        )

    return patch_tokens, n_patch


def sample_features_at_uv(feat_map, uv_ndc):
    """
    Bilinear-sample from *feat_map* at NDC [0,1] positions.

    Args:
        feat_map – [1, D, H, W]
        uv_ndc   – [M, 2]  (u, v) in [0, 1]

    Returns:
        [M, D]
    """
    grid = uv_ndc.clone()
    grid[:, 0] = grid[:, 0] * 2.0 - 1.0
    grid[:, 1] = grid[:, 1] * 2.0 - 1.0
    grid = grid.view(1, -1, 1, 2)                         # [1, M, 1, 2]
    sampled = F.grid_sample(
        feat_map, grid, mode="bilinear", align_corners=False
    )                                                      # [1, D, M, 1]
    return sampled.squeeze(0).squeeze(-1).T                # [M, D]


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def pca_to_rgb(features, valid_mask):
    """Reduce D-dim features to 3-dim via PCA → uint8 RGB."""
    feat_valid = features[valid_mask]
    if feat_valid.shape[0] < 4:
        return torch.zeros((features.shape[0], 3), dtype=torch.uint8)

    mean = feat_valid.mean(dim=0)
    centered = feat_valid - mean
    _, _, V = torch.pca_lowrank(centered, q=3)

    projected = (features - mean) @ V                      # [N, 3]
    proj_valid = projected[valid_mask]
    lo = proj_valid.quantile(0.02, dim=0)
    hi = proj_valid.quantile(0.98, dim=0)
    span = (hi - lo).clamp_min(1e-6)
    rgb = ((projected - lo) / span).clamp(0, 1)
    return (rgb * 255).to(torch.uint8)


def render_pca_image(uv_ndc, z, pca_rgb, visible_mask, resolution=512):
    """
    Scatter PCA colours onto a canvas with painter's-algorithm z-ordering.
    Far-to-near order so closer Gaussians overwrite farther ones.
    """
    vis_idx = torch.where(visible_mask)[0]
    u = (uv_ndc[vis_idx, 0] * resolution).long().clamp(0, resolution - 1)
    v = (uv_ndc[vis_idx, 1] * resolution).long().clamp(0, resolution - 1)
    z_vis = z[vis_idx]
    rgb_vis = pca_rgb[vis_idx]

    order = z_vis.argsort(descending=True)
    u, v, rgb_vis = u[order], v[order], rgb_vis[order]

    canvas = torch.zeros(resolution, resolution, 3, dtype=torch.uint8)
    canvas[v, u] = rgb_vis
    return canvas


def render_side_by_side(input_image, pca_canvas, resolution=512):
    """Return an [H, 2W, 3] uint8 array: input | PCA."""
    inp = input_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    inp_np = np.array(inp)
    pca_np = pca_canvas.numpy()
    return np.concatenate([inp_np, pca_np], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lift DINOv2 features to visible Gaussians (Step 2)"
    )
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--dino_image_size", type=int, default=518)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cupid_dir = os.path.join(args.output_dir, args.case_name, "cupid")
    vis_dir = os.path.join(args.output_dir, args.case_name, "gaussian_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # ---- load Step-1 outputs ------------------------------------------------
    print("[load] gaussians ...")
    gs = load_gaussians(os.path.join(cupid_dir, "gaussians.pt"))
    xyz = gs["xyz"]
    opacity = gs["opacity"]
    N = xyz.shape[0]
    print(f"[info] {N} gaussians")

    print("[load] pose ...")
    extrinsic, intrinsic = load_pose(os.path.join(cupid_dir, "pose.json"))

    img_path = os.path.join(cupid_dir, "input_masked.png")
    print(f"[load] image: {img_path}")
    image_rgb = load_image_rgb(img_path)

    # ---- project to image ---------------------------------------------------
    print("[project] gaussians -> image plane ...")
    uv_ndc, z = project_gaussians(xyz, extrinsic, intrinsic)
    visible_mask = compute_visibility_mask(uv_ndc, z, opacity)
    n_vis = visible_mask.sum().item()
    print(f"[info] {n_vis}/{N} visible ({n_vis / N * 100:.1f}%)")

    # ---- extract DINO features ----------------------------------------------
    print(f"[dino] {args.dino_model} @ {args.dino_image_size}px ...")
    feat_map, n_patch = extract_dino_features(
        image_rgb,
        model_name=args.dino_model,
        image_size=args.dino_image_size,
        device=args.device,
    )
    D = feat_map.shape[1]
    print(f"[info] feature dim={D}, patch grid={n_patch}x{n_patch}")

    # ---- sample features for visible gaussians ------------------------------
    print("[lift] sampling features at visible positions ...")
    feat_sem = torch.zeros((N, D), dtype=torch.float32)
    feat_visible = torch.zeros((N, 1), dtype=torch.float32)
    feat_valid = torch.zeros((N, 1), dtype=torch.float32)

    uv_vis = uv_ndc[visible_mask].to(args.device)
    sampled = sample_features_at_uv(feat_map, uv_vis)
    feat_sem[visible_mask] = sampled.cpu()
    feat_visible[visible_mask] = 1.0
    feat_valid[visible_mask] = 1.0

    # ---- save outputs -------------------------------------------------------
    print("[save] writing outputs ...")

    out = {
        "xyz": gs["xyz"],
        "scale": gs["scale"],
        "rotation": gs["rotation"],
        "opacity": gs["opacity"],
        "color": gs["color"],
        "aabb": gs["aabb"],
        "init_params": gs["init_params"],
        "feat_sem": feat_sem,
        "feat_visible": feat_visible,
        "feat_valid": feat_valid,
    }
    torch.save(out, os.path.join(vis_dir, "gaussians_vis.pt"))
    torch.save(visible_mask, os.path.join(vis_dir, "visible_mask.pt"))
    torch.save(feat_sem, os.path.join(vis_dir, "dino_feat_vis.pt"))
    print(f"  gaussians_vis.pt  ({N} pts, feat_dim={D})")
    print(f"  visible_mask.pt   ({n_vis} visible)")
    print(f"  dino_feat_vis.pt  ({D}-d features)")

    save_gaussians_ply(out, os.path.join(vis_dir, "gaussians_vis.ply"))
    print(f"  gaussians_vis.ply (standard 3DGS)")

    # ---- PCA visualisation --------------------------------------------------
    print("[viz] PCA render ...")
    pca_rgb = pca_to_rgb(feat_sem, visible_mask)
    pca_canvas = render_pca_image(
        uv_ndc, z, pca_rgb, visible_mask, resolution=args.render_resolution
    )
    sbs = render_side_by_side(image_rgb, pca_canvas, resolution=args.render_resolution)
    pca_path = os.path.join(vis_dir, "feature_render_pca.png")
    Image.fromarray(sbs).save(pca_path)
    print(f"  feature_render_pca.png ({sbs.shape[1]}x{sbs.shape[0]})")

    pca_colors_float = pca_rgb.float() / 255.0               # [N, 3] in [0,1]
    save_gaussians_ply(
        out, os.path.join(vis_dir, "gaussians_vis_pca.ply"),
        color_override=pca_colors_float,
    )
    print(f"  gaussians_vis_pca.ply (PCA-colored)")

    raw_path = os.path.join(vis_dir, "feature_render_raw.pt")
    torch.save(feat_map.cpu(), raw_path)
    print(f"  feature_render_raw.pt (DINO patch map)")

    print("[done]")


if __name__ == "__main__":
    main()
