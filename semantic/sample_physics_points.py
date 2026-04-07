import argparse
import os

import numpy as np
import open3d as o3d
from plyfile import PlyData


def load_points(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        pts = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path}: expected one array in npz, got {list(data.files)}")
        pts = data[data.files[0]]
    elif suffix == ".ply":
        ply = PlyData.read(path)
        v = ply["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 3:
        pts = pts[0]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected [N,3] or [T,N,3], got {pts.shape}")
    return pts


def main():
    parser = argparse.ArgumentParser(
        description="Sample a sparse spring-mass structure from dense object points"
    )
    parser.add_argument("--input", required=True, help="Dense points: .ply/.npy/.npz")
    parser.add_argument("--output", required=True, help="Output sparse .npy")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--voxel_size", type=float, default=0.003)
    args = parser.parse_args()

    pts = load_points(args.input)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)

    cur_n = len(pcd.points)
    if cur_n == 0:
        raise ValueError("No points left after voxel downsampling")
    if cur_n > args.num_points:
        pcd = pcd.farthest_point_down_sample(args.num_points)

    sampled = np.asarray(pcd.points, dtype=np.float32)
    np.save(args.output, sampled)

    print(f"[done] saved {args.output}")
    print(f"[info] input_points={pts.shape[0]} sampled_points={sampled.shape[0]}")


if __name__ == "__main__":
    main()
