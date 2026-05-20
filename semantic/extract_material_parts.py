"""
Extract material distribution and part segmentation for physics training (Step 5).

Uses Gaussian semantic feature clustering for part segmentation and VLM for material classification.

Pipeline:
1. Load gaussians with feat_sem from previous steps
2. Cluster valid gaussians by feat_sem (optional PCA -> K-means)
3. Render each cluster and save visualization
4. Use VLM to predict material distribution (full image as context + cluster crop)

Usage (from project root):
    python semantic/extract_material_parts.py \
        --case_name single_lift_cloth_4 \
        --output_dir results \
        --n_clusters 8

Input:
    results/[case_name]/cupid/input_masked.png
    results/[case_name]/cupid/pose.json
    results/[case_name]/gaussian_vis/gaussians_vis_sym.pt

Output:
    results/[case_name]/train/part_masks.pt
    results/[case_name]/train/part_features.pt
    results/[case_name]/train/material_distribution.pt
    results/[case_name]/train/material_embeddings.pt
    results/[case_name]/train/train_ready.pt
    results/[case_name]/train/parts_viz.png
    results/[case_name]/train/cluster_renders/  (per-cluster renderings)
"""

import argparse
import base64
import json
import os
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from physics_prior import (
    default_part_physics_prior,
    query_vlm_for_part_physics_prior,
)

# ---------------------------------------------------------------------------
# Default material classes (common physics-relevant materials)
# ---------------------------------------------------------------------------

# 10 common material classes (reduced from 20)
DEFAULT_MATERIALS = [
    "fabric",      # general textile (includes cloth, cotton, wool, felt, velvet)
    "leather",     # animal hide
    "rubber",      # rubber and foam
    "plastic",     # synthetic polymer
    "metal",       # metallic materials
    "wood",        # wooden materials
    "paper",       # paper and cardboard
    "silk",        # smooth fine textiles (includes nylon, polyester)
    "denim",       # stiff woven fabric (includes canvas)
    "fur",         # fur, hair, rope-like textures
]

# Material groups for physics parameter mapping
MATERIAL_GROUPS = {
    "soft_fabric": ["fabric", "silk"],
    "stiff_fabric": ["denim"],
    "leather": ["leather"],
    "rubber": ["rubber"],
    "plastic": ["plastic"],
    "metal": ["metal"],
    "wood": ["wood"],
    "paper": ["paper"],
    "fur": ["fur"],
}


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


def load_image_with_alpha(path):
    """Load image with alpha channel."""
    img = Image.open(path).convert("RGBA")
    return img


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

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
# Feature-based Clustering for Part Segmentation
# ---------------------------------------------------------------------------

def cluster_gaussians(
    feat_sem,
    feat_valid,
    n_clusters=5,
    use_pca=True,
    pca_dim=32,
    random_state=42,
):
    """
    Cluster valid gaussians by semantic features.

    Args:
        feat_sem: [N, D] semantic features
        feat_valid: [N, 1] or [N] validity mask
        n_clusters: number of clusters
        use_pca: whether to apply PCA before clustering
        pca_dim: target dimension for PCA
        random_state: random seed

    Returns:
        part_assignments: [N] int tensor, -1 for invalid gaussians
        cluster_centers: [K, D'] cluster centers (D' = pca_dim if use_pca else D)
        pca_model: fitted PCA model (or None)
    """
    N, D = feat_sem.shape

    # Get valid mask
    if feat_valid.dim() == 2:
        valid_mask = feat_valid.squeeze(-1) > 0.5
    else:
        valid_mask = feat_valid > 0.5

    valid_indices = torch.where(valid_mask)[0]
    n_valid = len(valid_indices)
    print(f"[cluster] {n_valid}/{N} valid gaussians for clustering")

    if n_valid == 0:
        return torch.full((N,), -1, dtype=torch.long), None, None

    # Extract valid features
    feat_valid_np = feat_sem[valid_indices].numpy()

    # Optional PCA
    pca_model = None
    if use_pca and D > pca_dim:
        print(f"[cluster] PCA: {D} -> {pca_dim}")
        pca_model = PCA(n_components=pca_dim, random_state=random_state)
        feat_reduced = pca_model.fit_transform(feat_valid_np)
    else:
        feat_reduced = feat_valid_np

    # Adjust n_clusters if needed
    actual_clusters = min(n_clusters, n_valid)
    if actual_clusters < n_clusters:
        print(f"[cluster] reducing n_clusters from {n_clusters} to {actual_clusters}")

    # K-means clustering
    print(f"[cluster] K-means with {actual_clusters} clusters ...")
    kmeans = KMeans(
        n_clusters=actual_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(feat_reduced)

    # Build full assignment array
    part_assignments = torch.full((N,), -1, dtype=torch.long)
    part_assignments[valid_indices] = torch.from_numpy(labels).long()

    return part_assignments, kmeans.cluster_centers_, pca_model


def compute_cluster_soft_assignments(feat_sem, cluster_centers, pca_model=None, temperature=0.1):
    """
    Compute soft assignment probabilities based on distance to cluster centers.

    Args:
        feat_sem: [N, D] semantic features
        cluster_centers: [K, D'] cluster centers
        pca_model: PCA model if used
        temperature: softmax temperature

    Returns:
        part_probs: [N, K] soft assignment probabilities
    """
    feat_np = feat_sem.numpy()

    if pca_model is not None:
        feat_np = pca_model.transform(feat_np)

    # Compute distances to cluster centers
    # [N, K]
    feat_t = torch.from_numpy(feat_np).float()
    centers_t = torch.from_numpy(cluster_centers).float()

    # Euclidean distance
    dists = torch.cdist(feat_t, centers_t)  # [N, K]

    # Convert to similarity (negative distance) and softmax
    similarities = -dists / temperature
    part_probs = F.softmax(similarities, dim=1)

    return part_probs


# ---------------------------------------------------------------------------
# Cluster Rendering
# ---------------------------------------------------------------------------

def render_cluster_2d(
    xyz,
    part_assignments,
    cluster_id,
    extrinsic,
    intrinsic,
    image_size,
    point_size=3,
    background_color=(255, 255, 255),
):
    """
    Simple 2D rendering of a cluster by projecting gaussians.

    Args:
        xyz: [N, 3] gaussian positions
        part_assignments: [N] cluster labels
        cluster_id: which cluster to render
        extrinsic: [4, 4] camera extrinsic
        intrinsic: [3, 3] camera intrinsic
        image_size: (W, H)
        point_size: radius of rendered points
        background_color: RGB tuple

    Returns:
        render: PIL Image
        mask: [H, W] boolean mask of rendered region
    """
    W, H = image_size
    mask = part_assignments == cluster_id

    if mask.sum() == 0:
        # Empty cluster
        img = Image.new("RGB", (W, H), background_color)
        return img, np.zeros((H, W), dtype=bool)

    # Project cluster points
    xyz_cluster = xyz[mask]
    uv_ndc, z = project_gaussians(xyz_cluster, extrinsic, intrinsic)

    # Filter visible points (in front of camera, within image)
    visible = (z > 0) & (uv_ndc[:, 0] >= 0) & (uv_ndc[:, 0] < 1) & \
              (uv_ndc[:, 1] >= 0) & (uv_ndc[:, 1] < 1)

    if visible.sum() == 0:
        img = Image.new("RGB", (W, H), background_color)
        return img, np.zeros((H, W), dtype=bool)

    uv_visible = uv_ndc[visible]
    z_visible = z[visible]

    # Convert to pixel coordinates
    u_px = (uv_visible[:, 0] * W).numpy()
    v_px = (uv_visible[:, 1] * H).numpy()

    # Create image
    img_arr = np.full((H, W, 3), background_color, dtype=np.uint8)
    render_mask = np.zeros((H, W), dtype=bool)

    # Sort by depth (far to near) for proper occlusion
    depth_order = np.argsort(-z_visible.numpy())

    # Assign colors based on depth (closer = brighter)
    z_np = z_visible.numpy()
    z_min, z_max = z_np.min(), z_np.max()
    if z_max > z_min:
        z_norm = (z_np - z_min) / (z_max - z_min)
    else:
        z_norm = np.ones_like(z_np) * 0.5

    for idx in depth_order:
        u, v = int(u_px[idx]), int(v_px[idx])
        # Draw a small circle
        for dy in range(-point_size, point_size + 1):
            for dx in range(-point_size, point_size + 1):
                if dx * dx + dy * dy <= point_size * point_size:
                    py, px = v + dy, u + dx
                    if 0 <= py < H and 0 <= px < W:
                        # Color based on depth
                        intensity = int(100 + 155 * (1 - z_norm[idx]))
                        img_arr[py, px] = (intensity, intensity, intensity)
                        render_mask[py, px] = True

    return Image.fromarray(img_arr), render_mask


def render_cluster_on_image(
    original_image,
    xyz,
    part_assignments,
    cluster_id,
    extrinsic,
    intrinsic,
    highlight_color=(255, 0, 0),
    alpha=0.5,
):
    """
    Render cluster overlay on original image.

    Args:
        original_image: PIL Image (RGB)
        xyz: [N, 3] gaussian positions
        part_assignments: [N] cluster labels
        cluster_id: which cluster to highlight
        extrinsic, intrinsic: camera params
        highlight_color: RGB tuple for cluster highlight
        alpha: blend alpha

    Returns:
        overlay: PIL Image with cluster highlighted
        mask_2d: [H, W] boolean mask of cluster region
        bbox: (x0, y0, x1, y1) bounding box of cluster
    """
    W, H = original_image.size
    mask = part_assignments == cluster_id

    if mask.sum() == 0:
        return original_image.copy(), np.zeros((H, W), dtype=bool), None

    # Project cluster points
    xyz_cluster = xyz[mask]
    uv_ndc, z = project_gaussians(xyz_cluster, extrinsic, intrinsic)

    # Filter visible
    visible = (z > 0) & (uv_ndc[:, 0] >= 0) & (uv_ndc[:, 0] < 1) & \
              (uv_ndc[:, 1] >= 0) & (uv_ndc[:, 1] < 1)

    if visible.sum() == 0:
        return original_image.copy(), np.zeros((H, W), dtype=bool), None

    uv_visible = uv_ndc[visible]
    u_px = (uv_visible[:, 0] * W).numpy()
    v_px = (uv_visible[:, 1] * H).numpy()

    # Create overlay and mask
    img_arr = np.array(original_image).copy()
    overlay = np.zeros_like(img_arr)
    mask_2d = np.zeros((H, W), dtype=bool)

    point_size = 2
    for u, v in zip(u_px, v_px):
        u, v = int(u), int(v)
        for dy in range(-point_size, point_size + 1):
            for dx in range(-point_size, point_size + 1):
                py, px = v + dy, u + dx
                if 0 <= py < H and 0 <= px < W:
                    overlay[py, px] = highlight_color
                    mask_2d[py, px] = True

    # Blend
    img_arr[mask_2d] = (
        (1 - alpha) * img_arr[mask_2d] + alpha * overlay[mask_2d]
    ).astype(np.uint8)

    # Compute bounding box
    x0, x1 = int(u_px.min()), int(u_px.max())
    y0, y1 = int(v_px.min()), int(v_px.max())

    return Image.fromarray(img_arr), mask_2d, (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# VLM-based Material Classification
# ---------------------------------------------------------------------------

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def query_vlm_for_material(
    full_image,
    cluster_image,
    cluster_id,
    materials,
    model_name=None,
):
    """
    Query local Ollama LLaVA for material distribution of a cluster.

    Args:
        full_image: PIL Image of full scene (context)
        cluster_image: PIL Image of cluster region (highlighted or cropped)
        cluster_id: cluster index for prompt
        materials: list of material names
        model_name: Ollama model name (default: llava:13b)

    Returns:
        distribution: [M] tensor of material probabilities
    """
    import requests

    M = len(materials)
    materials_str = ", ".join(materials)
    model = model_name or "llava:13b"

    prompt = f"""Look at these two images:
1. The full image showing an object
2. A cropped/highlighted region (cluster {cluster_id}) of the same object

For the highlighted region, estimate the probability distribution over these materials:
{materials_str}

Respond ONLY with a JSON object in this exact format:
{{"material_probs": {{{", ".join([f'"{m}": <probability>' for m in materials])}}}}}

The probabilities should sum to 1.0. Base your estimate on visual appearance (texture, sheen, flexibility, etc.)."""

    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"

    # Encode images
    full_b64 = image_to_base64(full_image)
    cluster_b64 = image_to_base64(cluster_image)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [full_b64, cluster_b64],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return _parse_vlm_response(result.get("response", ""), materials)
    except requests.exceptions.ConnectionError:
        print("[warn] Cannot connect to Ollama. Is it running? Start with: ollama serve")
        return torch.ones(M) / M
    except Exception as e:
        print(f"[warn] Ollama query failed: {e}, using uniform distribution")
        return torch.ones(M) / M


def _parse_vlm_response(response_text, materials):
    """Parse VLM response to extract material probabilities."""
    import re

    M = len(materials)

    # Try to extract JSON
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*"material_probs"[^{}]*\{[^{}]*\}[^{}]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            probs = data.get("material_probs", {})
        else:
            # Try parsing entire response as JSON
            data = json.loads(response_text)
            probs = data.get("material_probs", data)

        # Build distribution tensor
        distribution = torch.zeros(M)
        for i, mat in enumerate(materials):
            if mat in probs:
                distribution[i] = float(probs[mat])

        # Normalize
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        else:
            distribution = torch.ones(M) / M

        return distribution

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[warn] Failed to parse VLM response: {e}")
        print(f"  Response: {response_text[:200]}...")
        return torch.ones(M) / M


def compute_material_distribution_vlm(
    full_image,
    xyz,
    part_assignments,
    n_clusters,
    extrinsic,
    intrinsic,
    materials=None,
    model_name=None,
    save_dir=None,
):
    """
    Compute material distribution for each cluster using local Ollama LLaVA.

    Args:
        full_image: PIL Image
        xyz: [N, 3] gaussian positions
        part_assignments: [N] cluster labels
        n_clusters: number of clusters
        extrinsic, intrinsic: camera parameters
        materials: list of material names
        model_name: Ollama model name (default: llava:13b)
        save_dir: directory to save cluster renders (overlay + mask)

    Returns:
        distributions: [K, M] tensor of material probabilities
        material_names: list of material names
    """
    if materials is None:
        materials = DEFAULT_MATERIALS

    K = n_clusters
    M = len(materials)

    distributions = torch.zeros((K, M), dtype=torch.float32)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for k in range(K):
        count = (part_assignments == k).sum().item()
        if count == 0:
            print(f"  cluster {k}: empty, skipping")
            distributions[k] = torch.ones(M) / M
            continue

        print(f"  cluster {k}: {count} gaussians, querying LLaVA ...")

        # Render cluster overlay and mask
        overlay_img, mask_2d, bbox = render_cluster_on_image(
            full_image, xyz, part_assignments, k, extrinsic, intrinsic
        )

        # Save overlay and mask (uncropped)
        if save_dir:
            overlay_img.save(os.path.join(save_dir, f"cluster_{k:02d}_overlay.png"))
            mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode="L")
            mask_img.save(os.path.join(save_dir, f"cluster_{k:02d}_mask.png"))

        # Query LLaVA with overlay image (shows cluster in context)
        distribution = query_vlm_for_material(
            full_image,
            overlay_img,
            k,
            materials,
            model_name=model_name,
        )

        distributions[k] = distribution

    return distributions, materials


def compute_mixture_embeddings(distributions, embed_dim=64):
    """
    Compute learnable material mixture embeddings.

    For now, use a simple weighted sum of one-hot material vectors.
    In training, these would be replaced with learned embeddings.

    Args:
        distributions: [K, M] material probabilities
        embed_dim: embedding dimension

    Returns:
        embeddings: [K, embed_dim] mixture embeddings
    """
    K, M = distributions.shape

    # Simple projection: random but fixed material basis
    torch.manual_seed(42)
    material_basis = torch.randn(M, embed_dim)
    material_basis = material_basis / material_basis.norm(dim=1, keepdim=True)

    # Mixture embedding = weighted sum
    embeddings = distributions @ material_basis
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)

    return embeddings


# ---------------------------------------------------------------------------
# Part Feature Aggregation
# ---------------------------------------------------------------------------

def aggregate_part_features(feat_sem, part_assignments, K):
    """
    Aggregate gaussian semantic features per part.

    Args:
        feat_sem: [N, D] semantic features
        part_assignments: [N] part indices (-1 for unassigned)
        K: number of parts

    Returns:
        part_features: [K, D] aggregated features
        part_counts: [K] number of gaussians per part
    """
    N, D = feat_sem.shape
    part_features = torch.zeros((K, D), dtype=torch.float32)
    part_counts = torch.zeros(K, dtype=torch.long)

    for k in range(K):
        mask = part_assignments == k
        count = mask.sum().item()
        part_counts[k] = count
        if count > 0:
            part_features[k] = feat_sem[mask].mean(dim=0)

    return part_features, part_counts


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_clusters(
    image,
    xyz,
    part_assignments,
    n_clusters,
    extrinsic,
    intrinsic,
    distributions,
    materials,
    save_path,
):
    """Create visualization of clusters with material predictions."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    W, H = image.size
    K = n_clusters

    # Create colored overlay by projecting each cluster
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    for k in range(K):
        mask = part_assignments == k
        if mask.sum() == 0:
            continue

        xyz_cluster = xyz[mask]
        uv_ndc, z = project_gaussians(xyz_cluster, extrinsic, intrinsic)

        # Filter visible
        visible = (z > 0) & (uv_ndc[:, 0] >= 0) & (uv_ndc[:, 0] < 1) & \
                  (uv_ndc[:, 1] >= 0) & (uv_ndc[:, 1] < 1)

        if visible.sum() == 0:
            continue

        uv_visible = uv_ndc[visible]
        u_px = (uv_visible[:, 0] * W).long().clamp(0, W - 1)
        v_px = (uv_visible[:, 1] * H).long().clamp(0, H - 1)

        # Assign color based on cluster index
        hue = k / max(K, 1)
        color = hsv_to_rgb([hue, 0.7, 0.9])

        # Draw points
        point_size = 2
        for u, v in zip(u_px.numpy(), v_px.numpy()):
            for dy in range(-point_size, point_size + 1):
                for dx in range(-point_size, point_size + 1):
                    py, px = v + dy, u + dx
                    if 0 <= py < H and 0 <= px < W:
                        overlay[py, px, :3] = color
                        overlay[py, px, 3] = 0.6

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Clusters overlay
    axes[1].imshow(image)
    axes[1].imshow(overlay)
    axes[1].set_title(f"Semantic Clusters ({K} clusters)")
    axes[1].axis("off")

    # Material distribution
    if distributions is not None and len(distributions) > 0:
        # Show all materials with non-zero probability per cluster
        text_lines = []
        for k in range(min(K, 10)):
            if (part_assignments == k).sum() == 0:
                continue
            # Get all materials with probability > 0.01
            sorted_idx = distributions[k].argsort(descending=True)
            mats_str = []
            for i in sorted_idx:
                prob = distributions[k, i].item()
                if prob > 0.01:
                    mats_str.append(f"{materials[i]}:{prob:.2f}")
            text_lines.append(f"C{k}: {', '.join(mats_str)}")

        axes[2].text(0.05, 0.95, "\n".join(text_lines),
                     transform=axes[2].transAxes,
                     verticalalignment="top",
                     fontfamily="monospace",
                     fontsize=8)
        axes[2].set_title("Material Distribution")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract material distribution and parts (Step 5)"
    )
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")

    # Clustering parameters
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters for K-means")
    parser.add_argument("--use_pca", action="store_true", default=True,
                        help="Apply PCA before clustering")
    parser.add_argument("--no_pca", action="store_true",
                        help="Disable PCA")
    parser.add_argument("--pca_dim", type=int, default=32,
                        help="PCA target dimension")

    # Material parameters
    parser.add_argument("--materials", type=str, nargs="+", default=None,
                        help="Custom material list (default: common materials)")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Material embedding dimension")

    # LLaVA parameters
    parser.add_argument("--llava_model", type=str, default="llava:13b",
                        help="Ollama LLaVA model name (default: llava:13b)")
    parser.add_argument("--skip_vlm", action="store_true",
                        help="Skip LLaVA, use uniform distribution")
    parser.add_argument("--query_phys_prior", action="store_true",
                        help="Query VLM for physics priors and store them in train_ready.pt")
    parser.add_argument("--phys_llava_model", type=str, default=None,
                        help="Ollama model name for physics prior queries (default: same as --llava_model)")

    args = parser.parse_args()

    # Handle PCA flag
    use_pca = args.use_pca and not args.no_pca

    cupid_dir = os.path.join(args.output_dir, args.case_name, "cupid")
    vis_dir = os.path.join(args.output_dir, args.case_name, "gaussian_vis")
    train_dir = os.path.join(args.output_dir, args.case_name, "train")
    cluster_render_dir = os.path.join(train_dir, "cluster_renders")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(cluster_render_dir, exist_ok=True)

    # ---- load inputs ---------------------------------------------------------
    print("[load] gaussians_vis_sym.pt ...")
    gs_path = os.path.join(vis_dir, "gaussians_vis_sym.pt")
    if not os.path.exists(gs_path):
        gs_path = os.path.join(vis_dir, "gaussians_vis.pt")
        print(f"[warn] gaussians_vis_sym.pt not found, using gaussians_vis.pt")

    gs = load_gaussians(gs_path)
    xyz = gs["xyz"]
    feat_sem = gs["feat_sem"]
    feat_valid = gs.get("feat_valid", torch.ones((xyz.shape[0], 1)))
    N = xyz.shape[0]
    D = feat_sem.shape[1]
    print(f"[info] {N} gaussians, feat_dim={D}")

    print("[load] pose ...")
    extrinsic, intrinsic = load_pose(os.path.join(cupid_dir, "pose.json"))

    img_path = os.path.join(cupid_dir, "input_masked.png")
    print(f"[load] image: {img_path}")
    image_rgb = load_image_rgb(img_path)

    # ---- clustering on semantic features -------------------------------------
    print(f"[cluster] clustering gaussians by feat_sem (K={args.n_clusters}, PCA={use_pca}) ...")
    part_assignments, cluster_centers, pca_model = cluster_gaussians(
        feat_sem,
        feat_valid,
        n_clusters=args.n_clusters,
        use_pca=use_pca,
        pca_dim=args.pca_dim,
    )

    # Count actual clusters (some may be empty or merged)
    unique_labels = torch.unique(part_assignments[part_assignments >= 0])
    K = len(unique_labels)
    print(f"[info] {K} non-empty clusters")

    # Compute soft assignments
    if cluster_centers is not None:
        part_probs = compute_cluster_soft_assignments(
            feat_sem, cluster_centers, pca_model, temperature=0.1
        )
    else:
        part_probs = torch.zeros((N, K), dtype=torch.float32)

    assigned_count = (part_assignments >= 0).sum().item()
    print(f"[info] {assigned_count}/{N} gaussians assigned to clusters")

    # Print cluster sizes
    for k in range(K):
        count = (part_assignments == k).sum().item()
        print(f"  cluster {k}: {count} gaussians")

    # ---- material classification with VLM ------------------------------------
    materials = args.materials if args.materials else DEFAULT_MATERIALS

    if args.skip_vlm:
        print("[material] using uniform distribution ...")
        distributions = torch.ones((K, len(materials))) / len(materials)
    else:
        print(f"[material] querying LLaVA ({args.llava_model}) for material distribution ...")
        distributions, materials = compute_material_distribution_vlm(
            image_rgb,
            xyz,
            part_assignments,
            K,
            extrinsic,
            intrinsic,
            materials=materials,
            model_name=args.llava_model,
            save_dir=cluster_render_dir,
        )

    # Show top predictions
    print("[material] predictions:")
    for k in range(min(K, 10)):
        sorted_idx = distributions[k].argsort(descending=True)
        # Show all materials with probability > 0.01
        mats_parts = [f"{materials[i]}:{distributions[k,i]:.2f}"
                      for i in sorted_idx if distributions[k,i] > 0.01]
        print(f"  cluster {k}: {', '.join(mats_parts)}")

    # ---- physics prior with VLM -----------------------------------------------
    part_phys_prior = default_part_physics_prior(K)
    phys_prior_records = {"parts": []}

    if args.query_phys_prior and not args.skip_vlm:
        phys_model = args.phys_llava_model or args.llava_model
        print(f"[physics-prior] querying LLaVA ({phys_model}) for part-level physics priors ...")
        for k in range(K):
            overlay_img, _, _ = render_cluster_on_image(
                image_rgb, xyz, part_assignments, k, extrinsic, intrinsic
            )
            pred = query_vlm_for_part_physics_prior(
                image_rgb,
                overlay_img,
                k,
                model_name=phys_model,
            )
            part_phys_prior["part_phys_prior_mu"][k, 0] = float(pred["logk_mu"])
            part_phys_prior["part_phys_prior_log_sigma"][k, 0] = float(np.log(max(pred["logk_sigma"], 1e-4)))
            part_phys_prior["part_phys_prior_conf"][k, 0] = float(pred["confidence"])
            phys_prior_records["parts"].append({
                "part_id": k,
                "logk_mu": float(pred["logk_mu"]),
                "logk_sigma": float(pred["logk_sigma"]),
                "confidence": float(pred["confidence"]),
            })

    print("[physics-prior] part priors:")
    for k in range(min(K, 10)):
        print(
            f"  cluster {k}: mu={part_phys_prior['part_phys_prior_mu'][k,0].item():.3f}, "
            f"sigma={part_phys_prior['part_phys_prior_log_sigma'][k,0].exp().item():.3f}, "
            f"conf={part_phys_prior['part_phys_prior_conf'][k,0].item():.2f}"
        )

    # ---- compute embeddings --------------------------------------------------
    print("[embed] computing mixture embeddings ...")
    material_embeddings = compute_mixture_embeddings(distributions, args.embed_dim)

    # ---- aggregate part features ---------------------------------------------
    print("[aggregate] computing part-level semantic features ...")
    part_features, part_counts = aggregate_part_features(feat_sem, part_assignments, K)
    print(f"[info] part sizes: {part_counts.tolist()}")

    # ---- save outputs --------------------------------------------------------
    print("[save] writing outputs ...")

    # Part/cluster assignments
    part_data = {
        "part_assignments": part_assignments,      # [N] int, -1 for unassigned
        "part_probs": part_probs,                  # [N, K] soft assignments
        "num_parts": K,
        "cluster_centers": torch.from_numpy(cluster_centers) if cluster_centers is not None else None,
        "clustering_method": "kmeans",
        "use_pca": use_pca,
        "pca_dim": args.pca_dim if use_pca else None,
    }
    torch.save(part_data, os.path.join(train_dir, "part_masks.pt"))
    print(f"  part_masks.pt ({K} clusters)")

    # Part features
    part_feat_data = {
        "part_features": part_features,   # [K, D]
        "part_counts": part_counts,       # [K]
        "feat_dim": D,
    }
    torch.save(part_feat_data, os.path.join(train_dir, "part_features.pt"))
    print(f"  part_features.pt ({K} x {D})")

    # Material distribution
    mat_data = {
        "distributions": distributions,   # [K, M]
        "materials": materials,            # list of M names
        "material_groups": MATERIAL_GROUPS,
    }
    torch.save(mat_data, os.path.join(train_dir, "material_distribution.pt"))
    print(f"  material_distribution.pt ({K} x {len(materials)})")

    # Material embeddings
    embed_data = {
        "embeddings": material_embeddings,  # [K, embed_dim]
        "embed_dim": args.embed_dim,
    }
    torch.save(embed_data, os.path.join(train_dir, "material_embeddings.pt"))
    print(f"  material_embeddings.pt ({K} x {args.embed_dim})")

    phys_data = {
        **part_phys_prior,
        "records": phys_prior_records,
    }
    torch.save(phys_data, os.path.join(train_dir, "physics_prior.pt"))
    with open(os.path.join(train_dir, "physics_prior.json"), "w") as f:
        json.dump(phys_prior_records, f, indent=2)
    print("  physics_prior.pt / physics_prior.json")

    # Combined training-ready data
    train_ready = {
        # Gaussian data
        "xyz": gs["xyz"],
        "feat_sem": feat_sem,
        "feat_valid": feat_valid,
        "feat_sym_conf": gs.get("feat_sym_conf", torch.ones((N, 1))),

        # Part assignments
        "part_assignments": part_assignments,
        "part_probs": part_probs,
        "num_parts": K,

        # Part-level features
        "part_features": part_features,
        "part_counts": part_counts,

        # Material info
        "material_distributions": distributions,
        "material_embeddings": material_embeddings,
        "material_names": materials,

        # Physics priors from VLM
        "part_phys_prior_mu": part_phys_prior["part_phys_prior_mu"],
        "part_phys_prior_log_sigma": part_phys_prior["part_phys_prior_log_sigma"],
        "part_phys_prior_conf": part_phys_prior["part_phys_prior_conf"],
    }
    torch.save(train_ready, os.path.join(train_dir, "train_ready.pt"))
    print(f"  train_ready.pt (combined data)")

    # ---- visualization -------------------------------------------------------
    print("[viz] creating cluster visualization ...")
    viz_path = os.path.join(train_dir, "clusters_viz.png")
    visualize_clusters(
        image_rgb,
        xyz,
        part_assignments,
        K,
        extrinsic,
        intrinsic,
        distributions,
        materials,
        viz_path,
    )
    print(f"  clusters_viz.png")

    print("[done]")


if __name__ == "__main__":
    main()
