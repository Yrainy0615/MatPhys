import argparse
import glob
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def load_structure_points(final_data_path: str) -> np.ndarray:
    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    object_points = data["object_points"][0]
    other_surface_points = data["surface_points"]
    interior_points = data["interior_points"]
    structure_points = np.concatenate(
        [object_points, other_surface_points, interior_points], axis=0
    )
    return structure_points.astype(np.float32)


def load_camera_data(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    calibrate_path = os.path.join(case_dir, "calibrate.pkl")
    metadata_path = os.path.join(case_dir, "metadata.json")
    with open(calibrate_path, "rb") as f:
        c2ws = pickle.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    intrinsics = np.array(metadata["intrinsics"], dtype=np.float32)
    # Match train_models._load_case_cfg: Cupid-style normalized intrinsics
    # (fx,fy,cx,cy <= 2) are auto-scaled to pixel units.
    if np.max(np.abs(intrinsics[:, :2, :])) <= 2.0:
        wh = metadata.get("WH", None)
        if isinstance(wh, list) and len(wh) == 2 and not isinstance(wh[0], list):
            width, height = wh
        else:
            width, height = wh[0]
        intrinsics[:, 0, :] *= float(width)
        intrinsics[:, 1, :] *= float(height)
    w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws], dtype=np.float32)
    return w2cs, intrinsics


def _glob_images(path_prefix: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(path_prefix, "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(path_prefix, "*.jpg")))
    return paths


def load_images(color_dir: str, num_cams: int, frame_idx: int) -> List[np.ndarray]:
    """
    Supports both:
    - color/0.png, color/1.png, ...
    - color/<cam_idx>/<frame>.png
    Returns one image per camera index when possible.
    """
    images: List[np.ndarray] = []

    subdir_mode = all(os.path.isdir(os.path.join(color_dir, str(i))) for i in range(num_cams))
    if subdir_mode:
        for cam_idx in range(num_cams):
            cam_dir = os.path.join(color_dir, str(cam_idx))
            img_paths = _glob_images(cam_dir)
            if not img_paths:
                continue
            chosen_idx = min(frame_idx, len(img_paths) - 1)
            img = Image.open(img_paths[chosen_idx]).convert("RGB")
            images.append(np.array(img))
        return images

    flat_paths = _glob_images(color_dir)
    for path in flat_paths[:num_cams]:
        img = Image.open(path).convert("RGB")
        images.append(np.array(img))
    return images


def project_points_cv(points: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    cam = (w2c @ points_h.t()).t()[:, :3]
    z = cam[:, 2:3].clamp_min(1e-8)
    uv = (K @ cam.t()).t()[:, :2] / z
    return uv, z.squeeze(-1)


def extract_case_features(
    case_dir: str,
    final_data_path: str,
    model_name: str,
    image_size: int,
    device: str,
    frame_idx: int,
) -> np.ndarray:
    points_np = load_structure_points(final_data_path)
    w2cs_np, intrinsics_np = load_camera_data(case_dir)
    images_np = load_images(
        os.path.join(case_dir, "color"),
        num_cams=w2cs_np.shape[0],
        frame_idx=frame_idx,
    )
    if len(images_np) == 0:
        raise ValueError(f"No images found under: {os.path.join(case_dir, 'color')}")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    points = torch.from_numpy(points_np).to(device)
    w2cs = torch.from_numpy(w2cs_np).to(device)
    intrinsics = torch.from_numpy(intrinsics_np).to(device)

    n_patch = image_size // 14
    all_view_feats = []
    all_view_valid = []

    with torch.no_grad():
        for view_idx in range(min(len(images_np), w2cs.shape[0])):
            img_t = transform(Image.fromarray(images_np[view_idx])).unsqueeze(0).to(device)
            feats = model(img_t, is_training=True)
            patchtokens = feats["x_prenorm"][
                :, model.num_register_tokens + 1 :
            ].permute(0, 2, 1).reshape(1, -1, n_patch, n_patch)

            K = intrinsics[view_idx] if intrinsics.ndim == 3 else intrinsics
            uv_pix, z = project_points_cv(points, w2cs[view_idx], K)

            H, W = images_np[view_idx].shape[:2]
            uv_norm = uv_pix.clone()
            uv_norm[:, 0] = (uv_norm[:, 0] / max(W - 1, 1)) * 2.0 - 1.0
            uv_norm[:, 1] = (uv_norm[:, 1] / max(H - 1, 1)) * 2.0 - 1.0

            valid = (
                (uv_norm[:, 0] >= -1.0)
                & (uv_norm[:, 0] <= 1.0)
                & (uv_norm[:, 1] >= -1.0)
                & (uv_norm[:, 1] <= 1.0)
                & (z > 0)
            )

            grid = uv_norm.view(1, -1, 1, 2)
            sampled = F.grid_sample(
                patchtokens, grid, mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(-1).t()
            all_view_feats.append(sampled)
            all_view_valid.append(valid.float().unsqueeze(-1))

    feat_stack = torch.stack(all_view_feats, dim=0)
    valid_stack = torch.stack(all_view_valid, dim=0)
    feat_sum = (feat_stack * valid_stack).sum(dim=0)
    denom = valid_stack.sum(dim=0).clamp_min(1.0)
    feat_mean = feat_sum / denom
    return feat_mean.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Path containing case folders.")
    parser.add_argument("--cases", type=str, nargs="+", required=True, help="Case names.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="semantic/cache",
        help="Directory to save node semantic features.",
    )
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for case_name in args.cases:
        case_dir = os.path.join(args.base_path, case_name)
        final_data_path = os.path.join(case_dir, "final_data.pkl")
        node_sem = extract_case_features(
            case_dir=case_dir,
            final_data_path=final_data_path,
            model_name=args.model,
            image_size=args.image_size,
            device=args.device,
            frame_idx=args.frame_idx,
        )
        out_path = os.path.join(args.output_dir, f"{case_name}_node_sem.npz")
        np.savez_compressed(out_path, node_sem=node_sem)
        print(f"[saved] {out_path} shape={node_sem.shape}")


if __name__ == "__main__":
    main()
import argparse
import glob
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def load_structure_points(final_data_path: str) -> np.ndarray:
    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    object_points = data["object_points"][0]
    other_surface_points = data["surface_points"]
    interior_points = data["interior_points"]
    structure_points = np.concatenate(
        [object_points, other_surface_points, interior_points], axis=0
    )
    return structure_points.astype(np.float32)


def load_camera_data(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    calibrate_path = os.path.join(case_dir, "calibrate.pkl")
    metadata_path = os.path.join(case_dir, "metadata.json")
    with open(calibrate_path, "rb") as f:
        c2ws = pickle.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    intrinsics = np.array(metadata["intrinsics"], dtype=np.float32)
    # Match train_models._load_case_cfg: Cupid-style normalized intrinsics
    # (fx,fy,cx,cy <= 2) are auto-scaled to pixel units.
    if np.max(np.abs(intrinsics[:, :2, :])) <= 2.0:
        wh = metadata.get("WH", None)
        if isinstance(wh, list) and len(wh) == 2 and not isinstance(wh[0], list):
            width, height = wh
        else:
            width, height = wh[0]
        intrinsics[:, 0, :] *= float(width)
        intrinsics[:, 1, :] *= float(height)
    w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws], dtype=np.float32)
    return w2cs, intrinsics


def _glob_images(path_prefix: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(path_prefix, "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(path_prefix, "*.jpg")))
    return paths


def load_images(color_dir: str, num_cams: int, frame_idx: int) -> List[np.ndarray]:
    """
    Supports both:
    - color/0.png, color/1.png, ...
    - color/<cam_idx>/<frame>.png
    Returns one image per camera index when possible.
    """
    images: List[np.ndarray] = []

    subdir_mode = all(os.path.isdir(os.path.join(color_dir, str(i))) for i in range(num_cams))
    if subdir_mode:
        for cam_idx in range(num_cams):
            cam_dir = os.path.join(color_dir, str(cam_idx))
            img_paths = _glob_images(cam_dir)
            if not img_paths:
                continue
            chosen_idx = min(frame_idx, len(img_paths) - 1)
            img = Image.open(img_paths[chosen_idx]).convert("RGB")
            images.append(np.array(img))
        return images

    flat_paths = _glob_images(color_dir)
    for path in flat_paths[:num_cams]:
        img = Image.open(path).convert("RGB")
        images.append(np.array(img))
    return images


def project_points_cv(points: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    cam = (w2c @ points_h.t()).t()[:, :3]
    z = cam[:, 2:3].clamp_min(1e-8)
    uv = (K @ cam.t()).t()[:, :2] / z
    return uv, z.squeeze(-1)


def extract_case_features(
    case_dir: str,
    final_data_path: str,
    model_name: str,
    image_size: int,
    device: str,
    frame_idx: int,
) -> np.ndarray:
    points_np = load_structure_points(final_data_path)
    w2cs_np, intrinsics_np = load_camera_data(case_dir)
    images_np = load_images(
        os.path.join(case_dir, "color"),
        num_cams=w2cs_np.shape[0],
        frame_idx=frame_idx,
    )
    if len(images_np) == 0:
        raise ValueError(f"No images found under: {os.path.join(case_dir, 'color')}")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    points = torch.from_numpy(points_np).to(device)
    w2cs = torch.from_numpy(w2cs_np).to(device)
    intrinsics = torch.from_numpy(intrinsics_np).to(device)

    n_patch = image_size // 14
    all_view_feats = []
    all_view_valid = []

    with torch.no_grad():
        for view_idx in range(min(len(images_np), w2cs.shape[0])):
            img_t = transform(Image.fromarray(images_np[view_idx])).unsqueeze(0).to(device)
            feats = model(img_t, is_training=True)
            patchtokens = feats["x_prenorm"][
                :, model.num_register_tokens + 1 :
            ].permute(0, 2, 1).reshape(1, -1, n_patch, n_patch)

            K = intrinsics[view_idx] if intrinsics.ndim == 3 else intrinsics
            uv_pix, z = project_points_cv(points, w2cs[view_idx], K)

            H, W = images_np[view_idx].shape[:2]
            uv_norm = uv_pix.clone()
            uv_norm[:, 0] = (uv_norm[:, 0] / max(W - 1, 1)) * 2.0 - 1.0
            uv_norm[:, 1] = (uv_norm[:, 1] / max(H - 1, 1)) * 2.0 - 1.0

            valid = (
                (uv_norm[:, 0] >= -1.0)
                & (uv_norm[:, 0] <= 1.0)
                & (uv_norm[:, 1] >= -1.0)
                & (uv_norm[:, 1] <= 1.0)
                & (z > 0)
            )

            grid = uv_norm.view(1, -1, 1, 2)
            sampled = F.grid_sample(
                patchtokens, grid, mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(-1).t()
            all_view_feats.append(sampled)
            all_view_valid.append(valid.float().unsqueeze(-1))

    feat_stack = torch.stack(all_view_feats, dim=0)
    valid_stack = torch.stack(all_view_valid, dim=0)
    feat_sum = (feat_stack * valid_stack).sum(dim=0)
    denom = valid_stack.sum(dim=0).clamp_min(1.0)
    feat_mean = feat_sum / denom
    return feat_mean.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Path containing case folders.")
    parser.add_argument("--cases", type=str, nargs="+", required=True, help="Case names.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="semantic/cache",
        help="Directory to save node semantic features.",
    )
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for case_name in args.cases:
        case_dir = os.path.join(args.base_path, case_name)
        final_data_path = os.path.join(case_dir, "final_data.pkl")
        node_sem = extract_case_features(
            case_dir=case_dir,
            final_data_path=final_data_path,
            model_name=args.model,
            image_size=args.image_size,
            device=args.device,
            frame_idx=args.frame_idx,
        )
        out_path = os.path.join(args.output_dir, f"{case_name}_node_sem.npz")
        np.savez_compressed(out_path, node_sem=node_sem)
        print(f"[saved] {out_path} shape={node_sem.shape}")


if __name__ == "__main__":
    main()
