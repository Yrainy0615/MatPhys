import argparse
import csv
import json
import os
import shutil

base_path = "./data/different_types"
output_path = "./data/render_eval_data"
CONTROLLER_NAME = "hand"


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def copy_tree_contents(src_dir: str, dst_dir: str) -> None:
    ensure_dir(dst_dir)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def export_case(case_name: str) -> None:
    case_dir = os.path.join(base_path, case_name)
    if not os.path.exists(case_dir):
        return

    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    out_case_dir = os.path.join(output_path, case_name)
    ensure_dir(out_case_dir)
    ensure_dir(os.path.join(out_case_dir, "mask"))

    color_src = os.path.join(case_dir, "color")
    color_dst = os.path.join(out_case_dir, "color")
    if os.path.isdir(color_src):
        copy_tree_contents(color_src, color_dst)

    for i in range(3):
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        dst_mask_dir = os.path.join(out_case_dir, "mask", str(i))
        ensure_dir(dst_mask_dir)
        src_mask_dir = os.path.join(base_path, case_name, "mask", str(i), str(obj_idx))
        if os.path.isdir(src_mask_dir):
            copy_tree_contents(src_mask_dir, dst_mask_dir)

    shutil.copy2(
        os.path.join(base_path, case_name, "split.json"),
        os.path.join(out_case_dir, "split.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    args = parser.parse_args()

    ensure_dir(output_path)

    with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            case_name = row[0]
            if args.case_name is not None and case_name != args.case_name:
                continue
            export_case(case_name)


if __name__ == "__main__":
    main()
