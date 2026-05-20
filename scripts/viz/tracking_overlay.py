import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="data/ours_data")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def load_mask(path: Path, height: int, width: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def draw_points(frame, tracks_yx, visible, indices, color, radius):
    if len(indices) == 0:
        return
    h, w = frame.shape[:2]
    pts = tracks_yx[indices]
    vis = visible[indices]
    for (y, x), ok in zip(pts, vis):
        if not ok or not np.isfinite(x) or not np.isfinite(y):
            continue
        px = int(round(x))
        py = int(round(y))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(frame, (px, py), radius, color, -1, lineType=cv2.LINE_AA)


def write_overlay(video_path, tracks, visibility, object_idx, controller_idx, out_path, mode, fps, radius):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok or t >= tracks.shape[0]:
            break
        if mode in ("all", "object"):
            draw_points(frame, tracks[t], visibility[t], object_idx, (40, 220, 40), radius)
        if mode in ("all", "controller"):
            draw_points(frame, tracks[t], visibility[t], controller_idx, (30, 30, 255), radius + 1)
        writer.write(frame)
        t += 1
    cap.release()
    writer.release()
    return t


def main():
    args = parse_args()
    case_dir = Path(args.base_path) / args.case_name
    cam = args.camera_idx
    video_path = case_dir / "color" / f"{cam}.mp4"
    track_path = case_dir / "cotracker" / f"{cam}.npz"
    mask_info_path = case_dir / "mask" / f"mask_info_{cam}.json"

    data = np.load(track_path)
    tracks = data["tracks"].astype(np.float32)
    visibility = data["visibility"].astype(bool)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(mask_info_path) as f:
        mask_info = json.load(f)

    object_mask = None
    controller_mask = None
    controller_ids = []
    object_ids = []
    for key, name in mask_info.items():
        mask = load_mask(case_dir / "mask" / str(cam) / key / "0.png", height, width)
        if name == args.controller_name:
            controller_ids.append(int(key))
            controller_mask = mask if controller_mask is None else (controller_mask | mask)
        else:
            object_ids.append(int(key))
            object_mask = mask if object_mask is None else (object_mask | mask)

    y0 = np.clip(np.round(tracks[0, :, 0]).astype(int), 0, height - 1)
    x0 = np.clip(np.round(tracks[0, :, 1]).astype(int), 0, width - 1)
    object_idx = np.where(object_mask[y0, x0])[0] if object_mask is not None else np.array([], dtype=int)
    controller_idx = (
        np.where(controller_mask[y0, x0])[0] if controller_mask is not None else np.array([], dtype=int)
    )

    outputs = [
        ("all", case_dir / "final_data.mp4"),
        ("object", case_dir / "final_data_object_only.mp4"),
        ("controller", case_dir / "final_data_controller_only.mp4"),
    ]
    for mode, out_path in outputs:
        frames = write_overlay(
            video_path=video_path,
            tracks=tracks,
            visibility=visibility,
            object_idx=object_idx,
            controller_idx=controller_idx,
            out_path=out_path,
            mode=mode,
            fps=args.fps,
            radius=args.radius,
        )
        print(f"[write] {out_path} frames={frames}")

    print(
        f"[summary] T={tracks.shape[0]} total_tracks={tracks.shape[1]} "
        f"object_ids={object_ids} object_tracks={len(object_idx)} "
        f"controller_ids={controller_ids} controller_tracks={len(controller_idx)}"
    )


if __name__ == "__main__":
    main()
