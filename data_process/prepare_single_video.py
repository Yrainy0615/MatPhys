"""
Prepare a case directory from a single-view video.

Layout produced (mimicking the multi-camera baseline, with camera_idx=0 only):
    <base_path>/<case_name>/
        color/0/<frame>.png      # individual frames
        color/0.mp4              # re-encoded video at the chosen fps
        meta_video.json          # source video info (fps, frame_count, W, H)

Usage:
    python data_process/prepare_single_video.py \
        --video /path/to/input.mp4 \
        --base_path data/single_view \
        --case_name my_case \
        [--fps 30] [--max_frames 120]
"""
import argparse
import json
import os
import shutil
import subprocess

import imageio.v3 as iio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--base_path", required=True, help="Output base dir")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--fps", type=float, default=None,
                        help="Target fps. If None, keep source fps")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Truncate to first N frames after fps resampling")
    parser.add_argument("--start_sec", type=float, default=0.0)
    args = parser.parse_args()

    case_dir = os.path.join(args.base_path, args.case_name)
    color_dir = os.path.join(case_dir, "color", "0")
    os.makedirs(color_dir, exist_ok=True)

    meta = iio.immeta(args.video, plugin="FFMPEG")
    src_fps = float(meta.get("fps", 30.0))
    target_fps = float(args.fps) if args.fps is not None else src_fps

    frames = iio.imread(args.video, plugin="FFMPEG")  # [T, H, W, 3]
    if frames.ndim != 4:
        raise ValueError(f"Unexpected video shape {frames.shape}")
    T_src, H, W, _ = frames.shape

    start_frame = int(round(args.start_sec * src_fps))
    frames = frames[start_frame:]

    if abs(target_fps - src_fps) > 1e-6:
        step = src_fps / target_fps
        idx = np.arange(0, frames.shape[0], step).astype(int)
        idx = idx[idx < frames.shape[0]]
        frames = frames[idx]

    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    T = frames.shape[0]
    if T == 0:
        raise RuntimeError("No frames left after fps / max_frames filtering")

    for i in range(T):
        Image.fromarray(frames[i]).save(os.path.join(color_dir, f"{i}.png"))

    video_path = os.path.join(case_dir, "color", "0.mp4")
    iio.imwrite(video_path, frames, plugin="FFMPEG", fps=target_fps,
                codec="libx264", quality=8)

    info = {
        "source_video": os.path.abspath(args.video),
        "source_fps": src_fps,
        "target_fps": target_fps,
        "frame_num": int(T),
        "W": int(W),
        "H": int(H),
        "start_sec": args.start_sec,
    }
    with open(os.path.join(case_dir, "meta_video.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[prepare_single_video] wrote {T} frames -> {color_dir}")
    print(f"[prepare_single_video] re-encoded video -> {video_path}")
    print(f"[prepare_single_video] meta -> {os.path.join(case_dir, 'meta_video.json')}")


if __name__ == "__main__":
    main()
