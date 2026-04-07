"""
Propagate semantic features from visible to invisible Gaussians via symmetry.

Usage (from project root):
    python semantic/propagate_gaussian_symmetry.py \
        --case_name single_lift_sloth \
        --output_dir results

Input:
    results/[case_name]/gaussian_vis/gaussians_vis.pt

Output:
    results/[case_name]/gaussian_vis/gaussians_vis_sym.ply
    results/[case_name]/gaussian_vis/feature_render_sym_pca.png
    results/[case_name]/gaussian_vis/symmetry_plane.json
    results/[case_name]/gaussian_vis/symmetry_match.pt
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# PLY export (same as lift_dino_to_gaussian.py)
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x):
    return torch.log(x.clamp(1e-7, 1 - 1e-7) / (1 - x.clamp(1e-7, 1 - 1e-7)))


def save_gaussians_ply(gs, path, color_override=None):
    """Save Gaussian attributes as a 3DGS-compatible PLY."""
    xyz = gs["xyz"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    if color_override is not None:
        f_dc = color_override.detach().cpu().numpy()
    else:
        f_dc = gs["color"].detach().squeeze(1).cpu().numpy()

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


# ---------------------------------------------------------------------------
# Symmetry estimation
# ---------------------------------------------------------------------------

def estimate_symmetry_plane(xyz, n_candidates=3):
    """
    Estimate the best symmetry plane using PCA axes.

    Returns the plane with highest symmetry score (measured by point matching).
    Plane is represented as (normal, d) where normal.dot(x) = d is on plane.
    """
    xyz_np = xyz.detach().cpu().numpy()
    centroid = xyz_np.mean(axis=0)
    centered = xyz_np - centroid

    # PCA to get principal axes
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    best_plane = None
    best_score = -1

    # Test each principal axis as potential symmetry plane normal
    for i in range(min(n_candidates, 3)):
        normal = Vt[i]

        # Reflect points across plane through centroid
        # reflection: p' = p - 2 * ((p - c).dot(n)) * n
        proj = np.dot(centered, normal)[:, None] * normal[None, :]
        reflected = centered - 2 * proj
        reflected_pts = reflected + centroid

        # Score: mean nearest neighbor distance after reflection
        tree = cKDTree(xyz_np)
        dists, _ = tree.query(reflected_pts, k=1)
        score = 1.0 / (np.mean(dists) + 1e-6)

        if score > best_score:
            best_score = score
            best_plane = {
                "normal": normal.tolist(),
                "centroid": centroid.tolist(),
                "score": float(score),
                "axis_idx": i
            }

    return best_plane


def reflect_point(xyz, normal, centroid):
    """Reflect points across the symmetry plane."""
    normal = torch.tensor(normal, dtype=xyz.dtype, device=xyz.device)
    centroid = torch.tensor(centroid, dtype=xyz.dtype, device=xyz.device)

    centered = xyz - centroid
    proj = (centered @ normal).unsqueeze(-1) * normal.unsqueeze(0)
    reflected = centered - 2 * proj
    return reflected + centroid


def find_symmetric_correspondences(xyz, visible_mask, normal, centroid, max_dist=0.1):
    """
    For each invisible point, find its symmetric visible counterpart.

    Returns:
        match_idx: [N_invisible] indices into visible points, -1 if no match
        match_dist: [N_invisible] distances to matched points
    """
    visible_idx = torch.where(visible_mask)[0]
    invisible_idx = torch.where(~visible_mask)[0]

    if len(visible_idx) == 0 or len(invisible_idx) == 0:
        return torch.full((len(invisible_idx),), -1, dtype=torch.long), \
                torch.full((len(invisible_idx),), float('inf'))

    xyz_vis = xyz[visible_idx]
    xyz_invis = xyz[invisible_idx]

    # Reflect invisible points
    reflected_invis = reflect_point(xyz_invis, normal, centroid)

    # Find nearest visible point for each reflected invisible point
    tree = cKDTree(xyz_vis.numpy())
    dists, indices = tree.query(reflected_invis.numpy(), k=1)

    match_idx = torch.from_numpy(indices).long()
    match_dist = torch.from_numpy(dists).float()

    # Mark invalid matches
    invalid = match_dist > max_dist
    match_idx[invalid] = -1

    # Convert to global indices
    valid_mask = match_idx >= 0
    match_idx[valid_mask] = visible_idx[match_idx[valid_mask]]

    return match_idx, match_dist


def propagate_features(feat_sem, feat_valid, visible_mask, match_idx, confidence):
    """
    Propagate features from visible to invisible points using symmetry matches.
    """
    feat_sem_out = feat_sem.clone()
    feat_valid_out = feat_valid.clone()

    invisible_idx = torch.where(~visible_mask)[0]

    for i, invis_i in enumerate(invisible_idx):
        src_idx = match_idx[i].item()
        if src_idx >= 0:
            feat_sem_out[invis_i] = feat_sem[src_idx]
            feat_valid_out[invis_i] = confidence[i]

    return feat_sem_out, feat_valid_out


# ---------------------------------------------------------------------------
# PCA visualization (same as lift_dino_to_gaussian.py)
# ---------------------------------------------------------------------------

def pca_to_rgb(features, valid_mask):
    """Reduce D-dim features to 3-dim via PCA -> uint8 RGB."""
    feat_valid = features[valid_mask]
    if feat_valid.shape[0] < 4:
        return torch.zeros((features.shape[0], 3), dtype=torch.uint8)

    mean = feat_valid.mean(dim=0)
    centered = feat_valid - mean
    _, _, V = torch.pca_lowrank(centered, q=3)

    projected = (features - mean) @ V
    proj_valid = projected[valid_mask]
    lo = proj_valid.quantile(0.02, dim=0)
    hi = proj_valid.quantile(0.98, dim=0)
    span = (hi - lo).clamp_min(1e-6)
    rgb = ((projected - lo) / span).clamp(0, 1)
    return (rgb * 255).to(torch.uint8)


def render_pca_image(uv_ndc, z, pca_rgb, valid_mask, resolution=512):
    """Scatter PCA colours onto a canvas with z-ordering."""
    vis_idx = torch.where(valid_mask)[0]
    u = (uv_ndc[vis_idx, 0] * resolution).long().clamp(0, resolution - 1)
    v = (uv_ndc[vis_idx, 1] * resolution).long().clamp(0, resolution - 1)
    z_vis = z[vis_idx]
    rgb_vis = pca_rgb[vis_idx]

    order = z_vis.argsort(descending=True)
    u, v, rgb_vis = u[order], v[order], rgb_vis[order]

    canvas = torch.zeros(resolution, resolution, 3, dtype=torch.uint8)
    canvas[v, u] = rgb_vis
    return canvas


def project_gaussians(xyz, extrinsic, intrinsic):
    """Project 3D gaussian centres to 2D NDC coordinates."""
    N = xyz.shape[0]
    ones = torch.ones((N, 1), dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)
    cam = (extrinsic @ xyz_h.T).T[:, :3]
    z = cam[:, 2].clone()
    uv_h = (intrinsic @ cam.T).T
    uv_ndc = uv_h[:, :2] / uv_h[:, 2:3].clamp_min(1e-8)
    return uv_ndc, z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Propagate features via symmetry (Step 4)"
    )
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_match_dist", type=float, default=0.1,
                        help="Max distance for symmetry matching")
    parser.add_argument("--render_resolution", type=int, default=512)
    args = parser.parse_args()

    cupid_dir = os.path.join(args.output_dir, args.case_name, "cupid")
    vis_dir = os.path.join(args.output_dir, args.case_name, "gaussian_vis")

    # ---- load inputs ---------------------------------------------------------
    print("[load] gaussians_vis.pt ...")
    gs = load_gaussians(os.path.join(vis_dir, "gaussians_vis.pt"))
    xyz = gs["xyz"]
    feat_sem = gs["feat_sem"]
    feat_valid = gs["feat_valid"]
    feat_visible = gs["feat_visible"]
    visible_mask = feat_visible.squeeze(-1) > 0.5
    N = xyz.shape[0]
    D = feat_sem.shape[1]
    n_vis = visible_mask.sum().item()
    print(f"[info] {N} gaussians, {n_vis} visible, feat_dim={D}")

    # ---- estimate symmetry plane ---------------------------------------------
    print("[symmetry] estimating symmetry plane ...")
    plane = estimate_symmetry_plane(xyz)
    print(f"[info] best plane: axis={plane['axis_idx']}, score={plane['score']:.2f}")

    # ---- find correspondences ------------------------------------------------
    print("[symmetry] finding symmetric correspondences ...")
    match_idx, match_dist = find_symmetric_correspondences(
        xyz, visible_mask, plane["normal"], plane["centroid"],
        max_dist=args.max_match_dist
    )

    n_invis = (~visible_mask).sum().item()
    n_matched = (match_idx >= 0).sum().item()
    print(f"[info] {n_matched}/{n_invis} invisible points matched")

    # Compute confidence based on distance
    max_valid_dist = match_dist[match_idx >= 0].max() if n_matched > 0 else 1.0
    confidence = torch.zeros(n_invis)
    valid_match = match_idx >= 0
    confidence[valid_match] = 1.0 - (match_dist[valid_match] / (max_valid_dist + 1e-6)).clamp(0, 1)

    # ---- propagate features --------------------------------------------------
    print("[propagate] propagating features ...")
    feat_sem_sym, feat_valid_sym = propagate_features(
        feat_sem, feat_valid, visible_mask, match_idx, confidence
    )

    # Create symmetry confidence tensor
    feat_sym_conf = torch.zeros((N, 1), dtype=torch.float32)
    feat_sym_conf[visible_mask] = 1.0  # Visible points have full confidence
    invisible_idx = torch.where(~visible_mask)[0]
    feat_sym_conf[invisible_idx] = confidence.unsqueeze(-1)

    # ---- save outputs --------------------------------------------------------
    print("[save] writing outputs ...")

    # Update gaussian dict
    gs_out = {
        "xyz": gs["xyz"],
        "scale": gs["scale"],
        "rotation": gs["rotation"],
        "opacity": gs["opacity"],
        "color": gs["color"],
        "aabb": gs["aabb"],
        "init_params": gs["init_params"],
        "feat_sem": feat_sem_sym,
        "feat_visible": feat_visible,
        "feat_valid": feat_valid_sym,
        "feat_sym_conf": feat_sym_conf,
    }

    torch.save(gs_out, os.path.join(vis_dir, "gaussians_vis_sym.pt"))
    print(f"  gaussians_vis_sym.pt ({N} pts)")

    # Save symmetry info
    with open(os.path.join(vis_dir, "symmetry_plane.json"), "w") as f:
        json.dump(plane, f, indent=2)
    print(f"  symmetry_plane.json")

    # Save match info
    invisible_idx = torch.where(~visible_mask)[0]
    sym_match = {
        "invisible_idx": invisible_idx,
        "match_idx": match_idx,
        "match_dist": match_dist,
        "confidence": confidence,
    }
    torch.save(sym_match, os.path.join(vis_dir, "symmetry_match.pt"))
    print(f"  symmetry_match.pt ({n_matched} matches)")

    # ---- PCA visualization ---------------------------------------------------
    print("[viz] PCA render with symmetry features ...")

    # Load pose for projection
    extrinsic, intrinsic = load_pose(os.path.join(cupid_dir, "pose.json"))
    uv_ndc, z = project_gaussians(xyz, extrinsic, intrinsic)

    # All points with valid features (visible + matched invisible)
    all_valid_mask = feat_valid_sym.squeeze(-1) > 0.01

    pca_rgb = pca_to_rgb(feat_sem_sym, all_valid_mask)
    pca_canvas = render_pca_image(
        uv_ndc, z, pca_rgb, all_valid_mask, resolution=args.render_resolution
    )

    # Load original image for side-by-side
    img_path = os.path.join(cupid_dir, "input_masked.png")
    input_img = Image.open(img_path).convert("RGBA")
    bg = Image.new("RGBA", input_img.size, (0, 0, 0, 255))
    input_rgb = Image.alpha_composite(bg, input_img).convert("RGB")
    input_rgb = input_rgb.resize((args.render_resolution, args.render_resolution),
                                Image.Resampling.LANCZOS)
    inp_np = np.array(input_rgb)
    pca_np = pca_canvas.numpy()
    sbs = np.concatenate([inp_np, pca_np], axis=1)

    pca_path = os.path.join(vis_dir, "feature_render_sym_pca.png")
    Image.fromarray(sbs).save(pca_path)
    print(f"  feature_render_sym_pca.png ({sbs.shape[1]}x{sbs.shape[0]})")

    # Save PLY with PCA colors
    pca_colors_float = pca_rgb.float() / 255.0
    save_gaussians_ply(gs_out, os.path.join(vis_dir, "gaussians_vis_sym.ply"),
                        color_override=pca_colors_float)
    print(f"  gaussians_vis_sym.ply (PCA-colored)")

    print("[done]")


if __name__ == "__main__":
    main()