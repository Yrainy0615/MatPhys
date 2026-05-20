import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

prediction_dir = "./experiments"
base_path = "./data/different_types"
output_file = "results/final_results.csv"


def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    num_surface_points,
):
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)

    end_frame = min(end_frame, vertices.shape[0], object_points.shape[0], object_visibilities.shape[0])
    start_frame = min(start_frame, end_frame)

    for frame_idx in range(start_frame, end_frame):
        pred_points = vertices[frame_idx][:num_surface_points]
        gt_points = object_points[frame_idx][object_visibilities[frame_idx]]
        if gt_points.numel() == 0:
            continue
        chamfer_error = chamfer_distance(
            gt_points.unsqueeze(0),
            pred_points.unsqueeze(0),
            single_directional=True,
            norm=1,
        )[0]
        chamfer_errors.append(chamfer_error.item())

    return {
        "frame_len": len(chamfer_errors),
        "chamfer_error": float(np.mean(chamfer_errors)) if chamfer_errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Case Name",
                "Train Frame Num",
                "Train Chamfer Error",
                "Test Frame Num",
                "Test Chamfer Error",
            ]
        )

        dir_names = sorted(glob.glob(f"{prediction_dir}/*"))
        for dir_name in dir_names:
            case_name = os.path.basename(dir_name)
            if args.case_name is not None and case_name != args.case_name:
                continue

            inference_path = os.path.join(dir_name, "inference.pkl")
            case_dir = os.path.join(base_path, case_name)
            final_data_path = os.path.join(case_dir, "final_data.pkl")
            split_path = os.path.join(case_dir, "split.json")
            if not (os.path.isfile(inference_path) and os.path.isfile(final_data_path) and os.path.isfile(split_path)):
                continue

            print(f"Processing {case_name}")
            with open(inference_path, "rb") as f:
                vertices = pickle.load(f)
            with open(final_data_path, "rb") as f:
                data = pickle.load(f)
            with open(split_path, "r") as f:
                split = json.load(f)

            object_points = data["object_points"]
            object_visibilities = data["object_visibilities"]
            num_original_points = object_points.shape[1]
            num_surface_points = num_original_points + data["surface_points"].shape[0]
            train_frame = int(split["train"][1])
            test_frame = int(split["test"][1])

            results_train = evaluate_prediction(
                1,
                train_frame,
                vertices,
                object_points,
                object_visibilities,
                num_surface_points,
            )
            results_test = evaluate_prediction(
                train_frame,
                test_frame,
                vertices,
                object_points,
                object_visibilities,
                num_surface_points,
            )

            writer.writerow(
                [
                    case_name,
                    results_train["frame_len"],
                    results_train["chamfer_error"],
                    results_test["frame_len"],
                    results_test["chamfer_error"],
                ]
            )


if __name__ == "__main__":
    main()
