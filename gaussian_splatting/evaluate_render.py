import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


def scene_has_required_frames(output_scene_dir, frame_indices, view_idx):
    view_dir = os.path.join(output_scene_dir, str(view_idx))
    if not os.path.isdir(view_dir):
        return False
    return all(os.path.isfile(os.path.join(view_dir, f"{frame_idx:05d}.png")) for frame_idx in frame_indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--render_path", type=str, default="./data/render_eval_data")
    parser.add_argument("--human_mask_path", type=str, default="./data/different_types_human_mask")
    parser.add_argument("--output_dir", type=str, default="./gaussian_output_dynamic")
    args = parser.parse_args()

    render_path = args.render_path
    human_mask_path = args.human_mask_path
    output_dir = args.output_dir

    log_dir = "./results"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "output_dynamic.txt")

    with open(log_file_path, "w") as log_file:
        scene_names = sorted(os.listdir(render_path)) if os.path.isdir(render_path) else []
        if args.case_name is not None:
            scene_names = [scene for scene in scene_names if scene == args.case_name]

        all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = [], [], [], []
        all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []
        scene_metrics = {}

        for scene in scene_names:
            render_path_dir = os.path.join(render_path, scene)
            output_scene_dir = os.path.join(output_dir, scene)
            human_mask_dir = os.path.join(human_mask_path, scene)
            split_path = os.path.join(render_path_dir, "split.json")
            if not (os.path.isdir(render_path_dir) and os.path.isdir(human_mask_dir) and os.path.isfile(split_path)):
                continue

            with open(split_path, "r") as f:
                info = json.load(f)
            train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))
            test_f_idx_range = list(range(info["test"][0], info["test"][1]))

            if not scene_has_required_frames(output_scene_dir, train_f_idx_range + test_f_idx_range, view_idx=0):
                print(f"[skip] Missing rendered frames for {scene} in {output_scene_dir}")
                continue

            print("train indices range from", train_f_idx_range[0], "to", train_f_idx_range[-1])
            print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

            psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
            psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

            for view_idx in range(1):
                for frame_idx in train_f_idx_range:
                    gt = np.array(Image.open(os.path.join(render_path_dir, "color", str(view_idx), f"{frame_idx}.png")))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, "mask", str(view_idx), f"{frame_idx}.png")))
                    gt_mask = gt_mask.astype(np.float32) / 255.0

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f"{frame_idx:05d}.png")))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, "mask", str(view_idx), "0", f"{frame_idx}.png")))
                    inv_human_mask = (1.0 - human_mask / 255.0).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    gt[gt_mask == 0] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_train.append(psnr(render_tensor, gt_tensor).item())
                    ssims_train.append(ssim(render_tensor, gt_tensor).item())
                    lpipss_train.append(lpips(render_tensor, gt_tensor).item())
                    ious_train.append(compute_iou(gt_mask > 0, render_mask > 0))

                for frame_idx in test_f_idx_range:
                    gt = np.array(Image.open(os.path.join(render_path_dir, "color", str(view_idx), f"{frame_idx}.png")))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, "mask", str(view_idx), f"{frame_idx}.png")))
                    gt_mask = gt_mask.astype(np.float32) / 255.0

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f"{frame_idx:05d}.png")))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, "mask", str(view_idx), "0", f"{frame_idx}.png")))
                    inv_human_mask = (1.0 - human_mask / 255.0).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    gt[gt_mask == 0] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_test.append(psnr(render_tensor, gt_tensor).item())
                    ssims_test.append(ssim(render_tensor, gt_tensor).item())
                    lpipss_test.append(lpips(render_tensor, gt_tensor).item())
                    ious_test.append(compute_iou(gt_mask > 0, render_mask > 0))

            scene_metrics[scene] = {
                "psnr_train": float(np.mean(psnrs_train)) if psnrs_train else 0.0,
                "ssim_train": float(np.mean(ssims_train)) if ssims_train else 0.0,
                "lpips_train": float(np.mean(lpipss_train)) if lpipss_train else 0.0,
                "iou_train": float(np.mean(ious_train)) if ious_train else 0.0,
                "psnr_test": float(np.mean(psnrs_test)) if psnrs_test else 0.0,
                "ssim_test": float(np.mean(ssims_test)) if ssims_test else 0.0,
                "lpips_test": float(np.mean(lpipss_test)) if lpipss_test else 0.0,
                "iou_test": float(np.mean(ious_test)) if ious_test else 0.0,
            }

            all_psnrs_train.extend(psnrs_train)
            all_ssims_train.extend(ssims_train)
            all_lpipss_train.extend(lpipss_train)
            all_ious_train.extend(ious_train)
            all_psnrs_test.extend(psnrs_test)
            all_ssims_test.extend(ssims_test)
            all_lpipss_test.extend(lpipss_test)
            all_ious_test.extend(ious_test)

            print(f"===== Scene: {scene} =====")
            print(f"\t PSNR (train): {scene_metrics[scene]['psnr_train']:.4f}")
            print(f"\t SSIM (train): {scene_metrics[scene]['ssim_train']:.4f}")
            print(f"\t LPIPS (train): {scene_metrics[scene]['lpips_train']:.4f}")
            print(f"\t IoU (train): {scene_metrics[scene]['iou_train']:.4f}")
            print(f"\t PSNR (test): {scene_metrics[scene]['psnr_test']:.4f}")
            print(f"\t SSIM (test): {scene_metrics[scene]['ssim_test']:.4f}")
            print(f"\t LPIPS (test): {scene_metrics[scene]['lpips_test']:.4f}")
            print(f"\t IoU (test): {scene_metrics[scene]['iou_test']:.4f}")

        overall_psnr_train = float(np.mean(all_psnrs_train)) if all_psnrs_train else 0.0
        overall_ssim_train = float(np.mean(all_ssims_train)) if all_ssims_train else 0.0
        overall_lpips_train = float(np.mean(all_lpipss_train)) if all_lpipss_train else 0.0
        overall_iou_train = float(np.mean(all_ious_train)) if all_ious_train else 0.0
        overall_psnr_test = float(np.mean(all_psnrs_test)) if all_psnrs_test else 0.0
        overall_ssim_test = float(np.mean(all_ssims_test)) if all_ssims_test else 0.0
        overall_lpips_test = float(np.mean(all_lpipss_test)) if all_lpipss_test else 0.0
        overall_iou_test = float(np.mean(all_ious_test)) if all_ious_test else 0.0

        print("===== Overall Results Across All Scenes =====")
        print(f"\t Overall PSNR (train): {overall_psnr_train:.4f}")
        print(f"\t Overall SSIM (train): {overall_ssim_train:.4f}")
        print(f"\t Overall LPIPS (train): {overall_lpips_train:.4f}")
        print(f"\t Overall IoU (train): {overall_iou_train:.4f}")
        print(f"\t Overall PSNR (test): {overall_psnr_test:.4f}")
        print(f"\t Overall SSIM (test): {overall_ssim_test:.4f}")
        print(f"\t Overall LPIPS (test): {overall_lpips_test:.4f}")
        print(f"\t Overall IoU (test): {overall_iou_test:.4f}")

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("OVERALL RESULTS ACROSS ALL SCENES\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.write(f"Overall PSNR (train): {overall_psnr_train:.6f}\n")
        log_file.write(f"Overall SSIM (train): {overall_ssim_train:.6f}\n")
        log_file.write(f"Overall LPIPS (train): {overall_lpips_train:.6f}\n")
        log_file.write(f"Overall IoU (train): {overall_iou_train:.6f}\n\n")
        log_file.write(f"Overall PSNR (test): {overall_psnr_test:.6f}\n")
        log_file.write(f"Overall SSIM (test): {overall_ssim_test:.6f}\n")
        log_file.write(f"Overall LPIPS (test): {overall_lpips_test:.6f}\n")
        log_file.write(f"Overall IoU (test): {overall_iou_test:.6f}\n\n")
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("COMPACT METRICS TABLE BY SCENE\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.write(f"{'Scene':<50} | {'PSNR-train':<12} | {'SSIM-train':<12} | {'LPIPS-train':<14} | {'IoU-train':<12} | ")
        log_file.write(f"{'PSNR-test':<12} | {'SSIM-test':<12} | {'LPIPS-test':<14} | {'IoU-test':<12}\n")
        log_file.write("-" * 160 + "\n")
        for scene in scene_metrics:
            metrics = scene_metrics[scene]
            log_file.write(f"{scene[:50]:<50} | ")
            log_file.write(f"{metrics['psnr_train']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_train']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_train']:<14.6f} | ")
            log_file.write(f"{metrics['iou_train']:<12.6f} | ")
            log_file.write(f"{metrics['psnr_test']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_test']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_test']:<14.6f} | ")
            log_file.write(f"{metrics['iou_test']:<12.6f}\n")
        log_file.write("-" * 160 + "\n")
        log_file.write(f"{'OVERALL':<50} | ")
        log_file.write(f"{overall_psnr_train:<12.6f} | ")
        log_file.write(f"{overall_ssim_train:<12.6f} | ")
        log_file.write(f"{overall_lpips_train:<14.6f} | ")
        log_file.write(f"{overall_iou_train:<12.6f} | ")
        log_file.write(f"{overall_psnr_test:<12.6f} | ")
        log_file.write(f"{overall_ssim_test:<12.6f} | ")
        log_file.write(f"{overall_lpips_test:<14.6f} | ")
        log_file.write(f"{overall_iou_test:<12.6f}\n")
        print(f"\nMetrics have been saved to: {log_file_path}")


if __name__ == "__main__":
    main()
