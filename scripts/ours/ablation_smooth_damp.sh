#!/bin/bash
# Anti-vibration ablation on a single canonical case.
#
# Compares 4 cells, all on top of the damping-bug fix (predicted dashpot/drag now
# actually pushed into wp_dashpot/drag_damping):
#   A: bugfix-only      — new baseline
#   B: + acc_smooth     — warp-side jerk smoothness (cfg.acc_weight)
#   C: + damp_barrier   — torch-side soft floor on dashpot/drag
#   D: + both
#
# Sequential on a single GPU (default GPU 0). Each cell ~2-3h.

set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
CASE="${CASE:-single_push_sloth}"
EPOCHS="${EPOCHS:-200}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
ROOT="checkpoints/ablation_smooth_damp_${DATE_TAG}/${CASE}"
mkdir -p "$ROOT"

run_cell () {
    local TAG="$1" ; shift
    local SAVE_DIR="$ROOT/$TAG"
    mkdir -p "$SAVE_DIR"
    if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
        echo "[$(date)] [skip] $TAG already has last_checkpoint.pth"
        return
    fi
    echo "[$(date)] === Cell $TAG -> $SAVE_DIR ==="

    CUDA_VISIBLE_DEVICES="$GPU" python3 \
        semantic/train_model_video_material_simple.py \
        --case_name "$CASE" \
        --save_dir "$SAVE_DIR" \
        --base_path data/different_types \
        --experiments_dir data/different_types \
        --experiments_optimization_dir experiments_optimization \
        --case_to_material semantic/case_to_material_different_types.json \
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
        --epochs "$EPOCHS" \
        --eval_every 10 \
        --device cuda \
        --lr 3e-4 \
        --lambda_track 1.0 \
        --lambda_geo 1.0 \
        --lambda_render 0 \
        --grad_scale 1000.0 \
        --grad_clip 5.0 \
        --logk_residual_scale 1.0 \
        --logk_soft_clamp 0.25 \
        --lambda_phys_prior 0.001 \
        --phys_prior_part_mode empirical_kl \
        --phys_prior_global_mode barrier \
        --phys_prior_start_epoch 5 \
        --save_best_only \
        --vis_every 20 \
        "$@" \
        >> "$SAVE_DIR/train.log" 2>&1
    echo "[$(date)] === Done $TAG ==="
}

run_cell A_bugfix
run_cell B_accsmooth     --lambda_acc_smooth 0.01
run_cell C_dampbarrier   --lambda_damp_barrier 0.01 --min_dashpot 30 --min_drag 3
run_cell D_both          --lambda_acc_smooth 0.01 --lambda_damp_barrier 0.01 --min_dashpot 30 --min_drag 3

echo "[$(date)] All cells finished. Root: $ROOT"
