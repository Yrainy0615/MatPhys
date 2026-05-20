#!/usr/bin/env bash
#
# End-to-end training-data preparation for a single-view RGB video.
#
# Pipeline:
#   1. Extract frames (color/0/*.png, color/0.mp4)              [phystwin]
#   2. GroundedSAM2 video segmentation for object + hand        [phystwin]
#   3. Cupid 3D reconstruction of the object from frame 0       [cupid]
#   4. Co-Tracker dense tracking over the union mask            [phystwin]
#   5. Lift 2D tracks to 3D via ray-cast against Cupid mesh     [phystwin]
#   6. Lift DINOv2 features onto visible Gaussians              [phystwin]
#   7. Write final_data.pkl / metadata.json / calibrate.pkl /   [phystwin]
#      split.json (single-camera variant of the baseline)
#
# Usage:
#   bash scripts/preprocess/single_video.sh <video> <case_name> <category> <device_id> \
#       [base_path] [results_dir] [phystwin_env] [cupid_env]
#
# Example:
#   bash scripts/preprocess/single_video.sh /data/clip.mp4 my_sloth sloth 0
#
# Notes:
#   - <category> is the noun fed to GroundedSAM2 (e.g. "sloth", "zebra").
#     The TEXT_PROMPT becomes "<category>.hand".
#   - Defaults: base_path=data/single_view, results_dir=results,
#               phystwin_env=phystwin, cupid_env=cupid
#   - Optional env-var caps for prepare_single_video.py:
#       MAX_FRAMES=120  (truncate after fps resampling)
#       FPS=30          (target fps)
#       START_SEC=0     (skip the first N seconds)

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: bash scripts/preprocess/single_video.sh <video> <case_name> <category> <device_id> [base_path] [results_dir] [phystwin_env] [cupid_env]"
    exit 1
fi

VIDEO="$1"
CASE_NAME="$2"
CATEGORY="$3"
DEVICE_ID="$4"
BASE_PATH="${5:-data/single_view}"
RESULTS_DIR="${6:-results}"
PHYSTWIN_ENV="${7:-phystwin}"
CUPID_ENV="${8:-cupid}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CASE_DIR="${BASE_PATH}/${CASE_NAME}"
# Allow env-var override of the GroundedSAM2 prompt (e.g. PROMPT_OVERRIDE="plush monkey.hand")
TEXT_PROMPT="${PROMPT_OVERRIDE:-${CATEGORY}.hand}"

echo "=========================================================================="
echo "[run_single_video] case=${CASE_NAME}"
echo "  video      : ${VIDEO}"
echo "  category   : ${CATEGORY}"
echo "  prompt     : ${TEXT_PROMPT}"
echo "  device_id  : ${DEVICE_ID}"
echo "  base_path  : ${BASE_PATH}"
echo "  results    : ${RESULTS_DIR}"
echo "  phystwin   : ${PHYSTWIN_ENV}"
echo "  cupid      : ${CUPID_ENV}"
echo "=========================================================================="

PREP_EXTRA=()
if [ -n "${MAX_FRAMES:-}" ]; then PREP_EXTRA+=(--max_frames "${MAX_FRAMES}"); fi
if [ -n "${FPS:-}" ];         then PREP_EXTRA+=(--fps "${FPS}");                 fi
if [ -n "${START_SEC:-}" ];   then PREP_EXTRA+=(--start_sec "${START_SEC}");     fi

# ---- 1. Prepare frames -----------------------------------------------------
echo "[step 1/7] extract frames + re-encode video"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
    python data_process/prepare_single_video.py \
        --video "${VIDEO}" \
        --base_path "${BASE_PATH}" \
        --case_name "${CASE_NAME}" \
        "${PREP_EXTRA[@]}"

# ---- 2. GroundedSAM2 (single camera) ---------------------------------------
echo "[step 2/7] GroundedSAM2 video segmentation"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
    python data_process/segment_util_video.py \
        --base_path "${BASE_PATH}" \
        --case_name "${CASE_NAME}" \
        --TEXT_PROMPT "${TEXT_PROMPT}" \
        --camera_idx 0
rm -rf "${CASE_DIR}/tmp_data"

# ---- 3. Cupid reconstruction (cupid env) -----------------------------------
echo "[step 3/7] Cupid 3D reconstruction"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${CUPID_ENV}" \
    python semantic/run_cupid_case.py \
        --case_name "${CASE_NAME}" \
        --base_path "${BASE_PATH}" \
        --output_dir "${RESULTS_DIR}"

# ---- 4. Co-Tracker dense tracking (single camera) --------------------------
echo "[step 4/7] Co-Tracker dense tracking"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
    python data_process/dense_track_single.py \
        --base_path "${BASE_PATH}" \
        --case_name "${CASE_NAME}" \
        --camera_idx 0

# ---- 4b. Per-frame metric depth via Depth Anything (Cupid-scale anchor) ----
# Optional: skip with SKIP_DEPTH=1 to fall back to constant-z0 lift.
if [ "${SKIP_DEPTH:-0}" = "0" ]; then
    echo "[step 4b/7] Depth Anything per-frame depth + Cupid scale"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
        python data_process/predict_depth.py \
            --base_path "${BASE_PATH}" \
            --case_name "${CASE_NAME}" \
            --results_dir "${RESULTS_DIR}" \
            --cam_idx 0 \
            --device cuda
fi

# ---- 5. Lift 2D tracks to 3D via Cupid mesh (+ DA per-frame depth if available)
echo "[step 5/7] lift tracks to 3D"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
    python data_process/lift_tracks_single.py \
        --base_path "${BASE_PATH}" \
        --case_name "${CASE_NAME}" \
        --results_dir "${RESULTS_DIR}"

# ---- 6. Lift DINOv2 features ----------------------------------------------
echo "[step 6/7] lift DINOv2 features to Gaussians"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
    python semantic/lift_dino_to_gaussian.py \
        --case_name "${CASE_NAME}" \
        --output_dir "${RESULTS_DIR}" \
        --device cuda

# ---- 7. Write final_data.pkl / metadata / calibrate / split ----------------
echo "[step 7/7] write final dataset files"
OBJ_NPZ="${CASE_DIR}/object_tracks_3d.npz"
CTRL_NPZ="${CASE_DIR}/controller_tracks_3d.npz"

if [ ! -f "${CTRL_NPZ}" ]; then
    # Create an empty controller npz so prepare_final_data_case.py is happy.
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" python - <<PY
import numpy as np, os
T = np.load("${OBJ_NPZ}")["points"].shape[0]
np.savez(
    "${CTRL_NPZ}",
    points=np.zeros((T, 0, 3), dtype=np.float32),
    visibility=np.zeros((T, 0), dtype=bool),
)
PY
fi

PREP_FINAL_EXTRA=()
if [ -n "${TRAIN_FRAMES:-}" ]; then PREP_FINAL_EXTRA+=(--train_frames "${TRAIN_FRAMES}"); fi

CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${PHYSTWIN_ENV}" \
    python semantic/prepare_final_data_case.py \
        --base_path "${BASE_PATH}" \
        --results_dir "${RESULTS_DIR}" \
        --case_name "${CASE_NAME}" \
        --object_points "${OBJ_NPZ}" \
        --object_points_key points \
        --object_colors "${OBJ_NPZ}" \
        --object_colors_key colors \
        --object_visibilities "${OBJ_NPZ}" \
        --object_visibilities_key visibility \
        --object_motions_valid "${OBJ_NPZ}" \
        --object_motions_valid_key motions_valid \
        --controller_points "${CTRL_NPZ}" \
        --controller_points_key points \
        "${PREP_FINAL_EXTRA[@]}"

echo "=========================================================================="
echo "[run_single_video] DONE"
echo "  case dir : ${CASE_DIR}"
echo "  results  : ${RESULTS_DIR}/${CASE_NAME}"
echo "=========================================================================="
