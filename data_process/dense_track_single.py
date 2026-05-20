"""
Single-camera variant of data_process/dense_track.py.

Reads <base_path>/<case_name>/color/0.mp4 and the union of all masks under
<base_path>/<case_name>/mask/0/<obj_id>/0.png, samples up to N query pixels
inside that union, then runs Co-Tracker (online) to track them through the
whole clip.

Output:
    <base_path>/<case_name>/cotracker/0.npz
        tracks     : float32 [T, N, 2] stored as (row, col) i.e. (y, x)
        visibility : bool    [T, N]
    <base_path>/<case_name>/cotracker/0.mp4  (visualisation)
"""

import argparse
import glob
import os

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from utils.visualizer import Visualizer


def read_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--num_queries", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    case_dir = os.path.join(args.base_path, args.case_name)
    cam = args.camera_idx
    out_dir = os.path.join(case_dir, "cotracker")
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(case_dir, "color", f"{cam}.mp4")
    frames = iio.imread(video_path, plugin="FFMPEG")  # [T, H, W, 3]
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    T = frames.shape[0]
    print(f"[dense_track_single] video {video_path} -> T={T} HxW={frames.shape[1]}x{frames.shape[2]}")

    mask_paths = sorted(glob.glob(os.path.join(case_dir, "mask", str(cam), "*", "0.png")))
    if not mask_paths:
        raise FileNotFoundError(
            f"No first-frame masks under {os.path.join(case_dir, 'mask', str(cam))}/*/0.png"
        )
    mask = None
    for p in mask_paths:
        m = read_mask(p)
        mask = m if mask is None else np.logical_or(mask, m)
    print(f"[dense_track_single] union mask covers {int(mask.sum())} pixels")

    rng = np.random.default_rng(args.seed)
    yx = np.argwhere(mask)            # (Npix, 2) — rows, cols
    xy = yx[:, ::-1]                  # (Npix, 2) — x, y for cotracker
    if xy.shape[0] > args.num_queries:
        keep = rng.choice(xy.shape[0], size=args.num_queries, replace=False)
        xy = xy[keep]
    queries = np.concatenate([np.zeros((xy.shape[0], 1)), xy], axis=1)  # (N, 3) — t, x, y
    queries = torch.tensor(queries, dtype=torch.float32, device=device)[None]
    print(f"[dense_track_single] selected {queries.shape[1]} query points")

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    cotracker(video_chunk=video, is_first_step=True, queries=queries)
    for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, pred_visibility = cotracker(
            video_chunk=video[:, ind : ind + cotracker.step * 2]
        )

    vis = Visualizer(save_dir=out_dir, pad_value=0, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility, filename=f"{cam}")

    tracks_yx = pred_tracks[0].detach().cpu().numpy()[:, :, ::-1].astype(np.float32)
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)
    out_npz = os.path.join(out_dir, f"{cam}.npz")
    np.savez(out_npz, tracks=tracks_yx, visibility=visibility)
    print(f"[dense_track_single] saved {out_npz}  tracks={tracks_yx.shape} vis={visibility.shape}")


if __name__ == "__main__":
    main()
