#!/bin/bash
# Per-case video-network training with smooth regularizer (acc_smooth=0.01)
# AND --fit_all_frames so the trainer also fits the test split.
#
# Shards are LPT-balanced by full frame count across 5 GPUs (1, 4, 5, 6, 7).
# Max shard ~501 frames vs original round-robin 643 → ~22% wall-time saving.
#
# Each GPU runs its assigned cases sequentially in the background.

set -euo pipefail
cd "$(dirname "$0")/../.."

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
ROOT="checkpoints/per_case_smooth_fitall_${DATE_TAG}"
mkdir -p "$ROOT"

# LPT-balanced shards (frame counts shown in comment)
GPU1_CASES=(rope_double_hand single_clift_cloth_3 single_clift_cloth_1 single_push_rope)        # 499f
GPU4_CASES=(double_stretch_zebra single_lift_cloth_1 single_lift_dinosor double_lift_sloth weird_package)  # 495f
GPU5_CASES=(double_stretch_sloth double_lift_cloth_1 single_lift_sloth single_lift_zebra)       # 459f
GPU6_CASES=(single_lift_cloth double_lift_cloth_3 single_push_rope_1 single_push_sloth single_lift_rope)   # 501f
GPU7_CASES=(single_lift_cloth_4 single_lift_cloth_3 single_push_rope_4 double_lift_zebra)       # 469f

echo "[$(date)] root=$ROOT"

run_one_case () {
    local CASE="$1" ; local GPU="$2"
    local SAVE_DIR="$ROOT/$CASE"
    mkdir -p "$SAVE_DIR"
    if [ -f "$SAVE_DIR/last_checkpoint.pth" ]; then
        echo "[$(date)] [skip] $CASE (gpu $GPU) already has last_checkpoint.pth"
        return 0
    fi
    echo "[$(date)] === [gpu $GPU] start $CASE -> $SAVE_DIR ==="

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
        --epochs 200 \
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
        --lambda_acc_smooth 0.01 \
        --fit_all_frames \
        --save_best_only \
        --vis_every 20 \
        >> "$SAVE_DIR/train.log" 2>&1
    local rc=$?
    echo "[$(date)] === [gpu $GPU] done $CASE (rc=$rc) ==="
    return $rc
}

run_worker () {
    local GPU="$1" ; shift
    local WORKER_CASES=("$@")
    echo "[$(date)] [worker gpu $GPU] cases: ${WORKER_CASES[*]}"
    for CASE in "${WORKER_CASES[@]}"; do
        run_one_case "$CASE" "$GPU" || echo "[$(date)] [worker gpu $GPU] $CASE FAILED, continuing"
    done
    echo "[$(date)] [worker gpu $GPU] all done"
}

pids=()
launch_worker () {
    local GPU="$1" ; shift
    local LOG="$ROOT/worker_gpu${GPU}.log"
    ( run_worker "$GPU" "$@" ) >> "$LOG" 2>&1 &
    pids+=($!)
    echo "[$(date)] launched worker for gpu $GPU (pid=$!) -> $LOG ; cases=$*"
}

launch_worker 1 "${GPU1_CASES[@]}"
launch_worker 4 "${GPU4_CASES[@]}"
launch_worker 5 "${GPU5_CASES[@]}"
launch_worker 6 "${GPU6_CASES[@]}"
launch_worker 7 "${GPU7_CASES[@]}"

echo "[$(date)] all ${#pids[@]} workers launched. pids=${pids[*]}"
wait "${pids[@]}"
echo "[$(date)] all workers finished. root=$ROOT"
