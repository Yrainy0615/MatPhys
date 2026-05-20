import argparse
import shutil
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/render_gs_blend")
    parser.add_argument("--data_root", default="data/different_types")
    parser.add_argument("--cases", default="", help="Comma-separated case names. Empty means all cases under root.")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument(
        "--variants",
        default="ours,baseline",
        help="Comma-separated <name>.mp4 prefixes to extract (e.g. 'ours_global').",
    )
    parser.add_argument(
        "--copy_gt",
        action="store_true",
        default=True,
        help="Copy GT frames at sampled indices alongside the extracted variants.",
    )
    parser.add_argument(
        "--no_copy_gt",
        dest="copy_gt",
        action="store_false",
        help="Skip copying GT frames (useful when extracting an extra variant only).",
    )
    return parser.parse_args()


def midpoint_indices(num_total: int, num_frames: int) -> list[int]:
    return [min(num_total - 1, int((i + 0.5) * num_total / num_frames)) for i in range(num_frames)]


def read_video_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def extract_case(case_dir: Path, data_root: Path, num_frames: int,
                 variants: list[str], copy_gt: bool):
    case = case_dir.name
    gt_dir = data_root / case / "color" / "0"

    variant_paths = [(tag, case_dir / f"{tag}.mp4") for tag in variants]
    missing = [str(p) for _, p in variant_paths if not p.is_file()]
    if missing:
        return False, f"skip {case}: missing {missing}"
    if not gt_dir.is_dir():
        return False, f"skip {case}: missing GT dir {gt_dir}"

    gt_frames = sorted(gt_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(gt_frames) < num_frames:
        return False, f"skip {case}: only {len(gt_frames)} GT frames"

    indices = midpoint_indices(len(gt_frames), num_frames)
    for out_idx, frame_idx in enumerate(indices, start=1):
        for tag, video_path in variant_paths:
            frame = read_video_frame(video_path, frame_idx)
            out_path = case_dir / f"{tag}_{case}_{out_idx:02d}.png"
            cv2.imwrite(str(out_path), frame)

        if copy_gt:
            gt_src = gt_dir / f"{frame_idx}.png"
            gt_dst = case_dir / f"gt_{case}_{out_idx:02d}.png"
            shutil.copyfile(gt_src, gt_dst)

    return True, f"{case}: frames {indices}"


def main():
    args = parse_args()
    root = Path(args.root)
    data_root = Path(args.data_root)

    if args.cases:
        case_dirs = [root / c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    ok_count = 0
    for case_dir in case_dirs:
        ok, msg = extract_case(case_dir, data_root, args.num_frames, variants, args.copy_gt)
        print(msg)
        ok_count += int(ok)
    print(f"done: {ok_count}/{len(case_dirs)} cases")


if __name__ == "__main__":
    main()
