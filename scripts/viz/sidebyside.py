"""Compose side-by-side mp4: original RGB | 3DGS render | (optional) overlay alpha-blend."""
import os, sys, glob, argparse
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", required=True, help="dir containing 0.png 1.png ...")
    ap.add_argument("--gs_dir", required=True, help="dir containing 00000.png ...")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--with_overlay", action="store_true",
                    help="add 3rd panel: alpha-blend GS over RGB")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    gs_paths = sorted(glob.glob(os.path.join(args.gs_dir, "*.png")))
    if not gs_paths:
        raise SystemExit(f"no GS frames in {args.gs_dir}")
    n = len(gs_paths)

    rgb_paths = sorted(
        [p for p in glob.glob(os.path.join(args.rgb_dir, "*.png"))],
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if len(rgb_paths) < n:
        raise SystemExit(f"RGB has {len(rgb_paths)} frames, GS has {n}")

    sample_gs = cv2.imread(gs_paths[0])
    sample_rgb = cv2.imread(rgb_paths[0])
    if sample_rgb.shape != sample_gs.shape:
        sample_rgb = cv2.resize(sample_rgb, (sample_gs.shape[1], sample_gs.shape[0]))
    H, W = sample_gs.shape[:2]
    panels = 3 if args.with_overlay else 2
    sep_w = 4
    out_w = W * panels + sep_w * (panels - 1)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, args.fps, (out_w, H))

    for i, gs_path in enumerate(gs_paths):
        rgb = cv2.imread(rgb_paths[i])
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        gs = cv2.imread(gs_path)
        sep = np.full((H, sep_w, 3), 0, dtype=np.uint8)
        parts = [rgb, sep, gs]
        if args.with_overlay:
            blend = cv2.addWeighted(rgb, 1 - args.alpha, gs, args.alpha, 0)
            parts.extend([sep, blend])
        frame = np.concatenate(parts, axis=1)
        cv2.putText(frame, "GT RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "3DGS render", (W + sep_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if args.with_overlay:
            cv2.putText(frame, "overlay", (2 * (W + sep_w) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(frame)

    writer.release()
    print(f"saved {args.out_path}  ({n} frames, {out_w}x{H})")


if __name__ == "__main__":
    main()
