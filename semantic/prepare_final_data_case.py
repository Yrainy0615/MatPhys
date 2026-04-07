import argparse
import glob
import json
import os
import pickle

import numpy as np
from PIL import Image


def _load_array(path, key=None):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if key is not None:
            if key not in data:
                raise KeyError(f"{path}: key '{key}' not found. Available: {list(data.keys())}")
            return data[key]
        if len(data.files) != 1:
            raise ValueError(
                f"{path}: contains multiple arrays {list(data.files)}; pass --*_key to select one"
            )
        return data[data.files[0]]
    raise ValueError(f"Unsupported array format: {path}")


def _load_optional_array(path, key=None):
    if path is None:
        return None
    return _load_array(path, key=key)


def _first_image_size(color_dir):
    frame_paths = []
    for path in glob.glob(os.path.join(color_dir, "*.png")):
        name = os.path.basename(path)
        if name.startswith("._"):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".png" or not stem.isdigit():
            continue
        frame_paths.append(path)
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found under {color_dir}")
    frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    with Image.open(frame_paths[0]) as img:
        return img.size


def main():
    parser = argparse.ArgumentParser(
        description="Prepare final_data.pkl, split.json, metadata.json, and calibrate.pkl"
    )
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--base_path", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--object_points", required=True, help="[T,N,3] .npy or .npz")
    parser.add_argument("--object_points_key", default=None)
    parser.add_argument("--controller_points", required=True, help="[T,C,3] .npy or .npz")
    parser.add_argument("--controller_points_key", default=None)
    parser.add_argument("--object_colors", default=None, help="[T,N,3] or [N,3] .npy/.npz")
    parser.add_argument("--object_colors_key", default=None)
    parser.add_argument("--object_visibilities", default=None, help="[T,N] bool .npy/.npz")
    parser.add_argument("--object_visibilities_key", default=None)
    parser.add_argument("--object_motions_valid", default=None, help="[T,N] bool .npy/.npz")
    parser.add_argument("--object_motions_valid_key", default=None)
    parser.add_argument("--surface_points", default=None, help="[Ns,3] .npy/.npz")
    parser.add_argument("--surface_points_key", default=None)
    parser.add_argument("--interior_points", default=None, help="[Ni,3] .npy/.npz")
    parser.add_argument("--interior_points_key", default=None)
    parser.add_argument("--pose_json", default=None, help="Defaults to results/<case>/cupid/pose.json")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--train_frames", type=int, default=None)
    args = parser.parse_args()

    case_dir = os.path.join(args.base_path, args.case_name)
    os.makedirs(case_dir, exist_ok=True)

    object_points = _load_array(args.object_points, key=args.object_points_key).astype(np.float32)
    controller_points = _load_array(
        args.controller_points, key=args.controller_points_key
    ).astype(np.float32)

    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(f"object_points must have shape [T,N,3], got {object_points.shape}")
    if controller_points.ndim != 3 or controller_points.shape[-1] != 3:
        raise ValueError(
            f"controller_points must have shape [T,C,3], got {controller_points.shape}"
        )
    if object_points.shape[0] != controller_points.shape[0]:
        raise ValueError(
            f"Frame count mismatch: object_points {object_points.shape[0]} vs "
            f"controller_points {controller_points.shape[0]}"
        )

    num_frames, num_points = object_points.shape[:2]

    object_colors = _load_optional_array(args.object_colors, key=args.object_colors_key)
    if object_colors is None:
        object_colors = np.zeros((num_frames, num_points, 3), dtype=np.float32)
    else:
        object_colors = object_colors.astype(np.float32)
        if object_colors.ndim == 2 and object_colors.shape == (num_points, 3):
            object_colors = np.repeat(object_colors[None, ...], num_frames, axis=0)
        if object_colors.shape != (num_frames, num_points, 3):
            raise ValueError(
                f"object_colors must have shape [T,N,3] or [N,3], got {object_colors.shape}"
            )

    object_visibilities = _load_optional_array(
        args.object_visibilities, key=args.object_visibilities_key
    )
    if object_visibilities is None:
        object_visibilities = np.ones((num_frames, num_points), dtype=bool)
    else:
        object_visibilities = object_visibilities.astype(bool)
        if object_visibilities.shape != (num_frames, num_points):
            raise ValueError(
                f"object_visibilities must have shape [T,N], got {object_visibilities.shape}"
            )

    object_motions_valid = _load_optional_array(
        args.object_motions_valid, key=args.object_motions_valid_key
    )
    if object_motions_valid is None:
        object_motions_valid = np.zeros((num_frames, num_points), dtype=bool)
        if num_frames > 1:
            object_motions_valid[:-1] = (
                object_visibilities[:-1] & object_visibilities[1:]
            )
    else:
        object_motions_valid = object_motions_valid.astype(bool)
        if object_motions_valid.shape != (num_frames, num_points):
            raise ValueError(
                f"object_motions_valid must have shape [T,N], got {object_motions_valid.shape}"
            )

    surface_points = _load_optional_array(args.surface_points, key=args.surface_points_key)
    interior_points = _load_optional_array(args.interior_points, key=args.interior_points_key)
    if surface_points is None:
        surface_points = np.zeros((0, 3), dtype=np.float32)
    else:
        surface_points = surface_points.astype(np.float32).reshape(-1, 3)
    if interior_points is None:
        interior_points = np.zeros((0, 3), dtype=np.float32)
    else:
        interior_points = interior_points.astype(np.float32).reshape(-1, 3)

    pose_json = args.pose_json or os.path.join(
        args.results_dir, args.case_name, "cupid", "pose.json"
    )
    with open(pose_json, "r") as f:
        pose = json.load(f)
    intrinsic = np.asarray(pose["intrinsic"], dtype=np.float32)
    extrinsic = np.asarray(pose["extrinsic"], dtype=np.float32)
    c2w = np.linalg.inv(extrinsic).astype(np.float32)

    width = args.width
    height = args.height
    if width is None or height is None:
        color_dir = os.path.join(case_dir, "color", "0")
        width, height = _first_image_size(color_dir)

    train_frames = args.train_frames if args.train_frames is not None else num_frames

    final_data = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": object_visibilities,
        "object_motions_valid": object_motions_valid,
        "controller_points": controller_points,
        "surface_points": surface_points,
        "interior_points": interior_points,
    }
    with open(os.path.join(case_dir, "final_data.pkl"), "wb") as f:
        pickle.dump(final_data, f)

    split = {"train": [0, int(train_frames)], "test": [0, int(num_frames)]}
    with open(os.path.join(case_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    metadata = {
        "WH": [[int(width), int(height)]],
        "intrinsics": [intrinsic.tolist()],
    }
    with open(os.path.join(case_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(case_dir, "calibrate.pkl"), "wb") as f:
        pickle.dump(np.stack([c2w], axis=0), f)

    print(f"[done] wrote {os.path.join(case_dir, 'final_data.pkl')}")
    print(f"[done] wrote {os.path.join(case_dir, 'split.json')}")
    print(f"[done] wrote {os.path.join(case_dir, 'metadata.json')}")
    print(f"[done] wrote {os.path.join(case_dir, 'calibrate.pkl')}")
    print(f"[info] frames={num_frames}, object_points={num_points}, controller_points={controller_points.shape[1]}")
    print(f"[info] surface_points={surface_points.shape[0]}, interior_points={interior_points.shape[0]}")


if __name__ == "__main__":
    main()
