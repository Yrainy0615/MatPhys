#!/usr/bin/env bash
#
# Run the current "ours" (per-case video material) training on ONE case.
# Mirrors scripts/ours/train_all.sh's args, but parameterised so
# you can point base_path / case_to_material at a fresh single-video case
# produced by scripts/preprocess/single_video.sh.
#
# Usage:
#   bash scripts/ours/train_single.sh <case_name> <device_id> \
#       [base_path] [case_to_material] [save_root] [epochs]
#
# Example (single monocular video case):
#   bash scripts/ours/train_single.sh monkey 0 \
#       data/ours_data semantic/case_to_material_monkey_only.json \
#       checkpoints/monkey_ours_test 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/ours/train_single.sh <case_name> <device_id> [base_path] [case_to_material] [save_root] [epochs]"
    exit 1
fi

CASE_NAME="$1"
DEVICE_ID="$2"
BASE_PATH="${3:-data/different_types}"
CASE_TO_MATERIAL="${4:-semantic/case_to_material_different_types.json}"
SAVE_ROOT="${5:-checkpoints/ours_single_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${6:-50}"

SAVE_DIR="${SAVE_ROOT}/${CASE_NAME}"
mkdir -p "$SAVE_DIR"

echo "[$(date)] === ours single-case ==="
echo "  case             : ${CASE_NAME}"
echo "  device_id        : ${DEVICE_ID}"
echo "  base_path        : ${BASE_PATH}"
echo "  case_to_material : ${CASE_TO_MATERIAL}"
echo "  save_dir         : ${SAVE_DIR}"
echo "  epochs           : ${EPOCHS}"

CUDA_VISIBLE_DEVICES="${DEVICE_ID}" python3 \
    semantic/train_model_video_material_simple.py \
    --case_name "${CASE_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --base_path "${BASE_PATH}" \
    --experiments_dir "${BASE_PATH}" \
    --experiments_optimization_dir experiments_optimization \
    --case_to_material "${CASE_TO_MATERIAL}" \
    --results_dir results \
    --sem_cache_dir semantic/cache \
    --gaussian_root gaussian_output \
    --videomae_model MCG-NJU/videomae-base \
    --videomae_image_size 224 \
    --num_video_frames 16 \
    --d_motion 128 \
    --mat_codebook_dim 32 \
    --hidden_dim 256 \
    --num_materials 10 \
    --batch_size 4 \
    --num_workers 0 \
    --train_ratio 1.0 \
    --object_radius "${OBJECT_RADIUS:-0.02}" \
    --controller_radius "${CONTROLLER_RADIUS:-0.04}" \
    --epochs "${EPOCHS}" \
    --eval_every 10 \
    --device cuda \
    --lr 3e-4 \
    --lambda_track ${LAMBDA_TRACK:-1.0} \
    --lambda_geo ${LAMBDA_GEO:-1.0} \
    --lambda_render ${LAMBDA_RENDER:-1e-3} \
    --grad_scale 1000.0 \
    --grad_clip 5.0 \
    --logk_residual_scale 1.0 \
    --logk_soft_clamp 0.25 \
    --lambda_phys_prior 0.001 \
    --phys_prior_part_mode empirical_kl \
    --phys_prior_global_mode barrier \
    --phys_prior_start_epoch 5 \
    --lambda_acc_smooth 0.01 \
    --fit_all_frames \
    --save_best_only \
    --vis_every 20 \
    2>&1 | tee "${SAVE_DIR}/train.log"

echo "[$(date)] === done $CASE_NAME ==="
