import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
from scipy.spatial import KDTree

base_path = "./data/different_types"
prediction_path = "experiments"
output_file = "results/final_track.csv"


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    end_frame = min(end_frame, len(vertices), len(gt_track_3d))
    start_frame = min(start_frame, end_frame)
    for frame_idx in range(start_frame, end_frame):
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0.0
        else:
            track_error = float(np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1)))
        track_errors.append(track_error)
    return float(np.mean(track_errors)) if track_errors else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    args = parser.parse_args()

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Case Name", "Train Track Error", "Test Track Error"])

        dir_names = sorted(glob.glob(f"{base_path}/*"))
        for dir_name in dir_names:
            case_name = os.path.basename(dir_name)
            if args.case_name is not None and case_name != args.case_name:
                continue

            split_path = os.path.join(base_path, case_name, "split.json")
            inference_path = os.path.join(prediction_path, case_name, "inference.pkl")
            gt_track_path = os.path.join(base_path, case_name, "gt_track_3d.pkl")
            if not (os.path.isfile(split_path) and os.path.isfile(inference_path) and os.path.isfile(gt_track_path)):
                continue

            print(f"Processing {case_name}!!!!!!!!!!!!!!!")
            with open(split_path, "r") as f:
                split = json.load(f)
            train_frame = int(split["train"][1])
            test_frame = int(split["test"][1])
            with open(inference_path, "rb") as f:
                vertices = pickle.load(f)
            with open(gt_track_path, "rb") as f:
                gt_track_3d = pickle.load(f)

            mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
            _, idx = KDTree(vertices[0]).query(gt_track_3d[0][mask])

            train_track_error = evaluate_prediction(1, train_frame, vertices, gt_track_3d, idx, mask)
            test_track_error = evaluate_prediction(train_frame, test_frame, vertices, gt_track_3d, idx, mask)
            writer.writerow([case_name, train_track_error, test_track_error])


if __name__ == "__main__":
    main()
