#!/bin/bash
# Lift DINOv2 features to visible Gaussians (Step 2).
#
# Usage:
#   bash scripts/run_lift_dino.sh <device_id>
#   bash scripts/run_lift_dino.sh <device_id> [output_dir] [conda_env]
#   bash scripts/run_lift_dino.sh <case_name> <device_id>
#   bash scripts/run_lift_dino.sh <case_name> <device_id> [output_dir] [conda_env]
#
# Examples:
#   bash scripts/run_lift_dino.sh 2
#   bash scripts/run_lift_dino.sh 2 results phystwin
#   bash scripts/run_lift_dino.sh single_lift_cloth_4 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_lift_dino.sh <device_id> [output_dir] [conda_env]"
    echo "   or: bash scripts/run_lift_dino.sh <case_name> <device_id> [output_dir] [conda_env]"
    exit 1
fi

CASE_NAME=""
if [[ "$1" =~ ^[0-9]+$ ]]; then
    DEVICE_ID="$1"
    OUTPUT_DIR="${2:-results}"
    CONDA_ENV="${3:-phystwin}"
else
    if [ $# -lt 2 ]; then
        echo "Single-case mode requires a device id."
        echo "Usage: bash scripts/run_lift_dino.sh <case_name> <device_id> [output_dir] [conda_env]"
        exit 1
    fi
    CASE_NAME="$1"
    DEVICE_ID="$2"
    OUTPUT_DIR="${3:-results}"
    CONDA_ENV="${4:-phystwin}"
fi

run_case() {
    local case_name="$1"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${CONDA_ENV}" \
        python semantic/lift_dino_to_gaussian.py \
        --case_name "${case_name}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda
}

if [ -n "$CASE_NAME" ]; then
    echo "=== Step 2: Lift DINO features to Gaussians ==="
    echo "  case:      ${CASE_NAME}"
    echo "  device_id: ${DEVICE_ID}"
    echo "  output:    ${OUTPUT_DIR}"
    echo "  env:       ${CONDA_ENV}"
    echo ""
    run_case "$CASE_NAME"
else
    echo "=== Step 2: Lift DINO features to Gaussians (all cases) ==="
    echo "  device_id: ${DEVICE_ID}"
    echo "  output:    ${OUTPUT_DIR}"
    echo "  env:       ${CONDA_ENV}"
    echo ""
    count=0
    for cupid_dir in "${OUTPUT_DIR}"/*/cupid; do
        [ -f "${cupid_dir}/gaussians.pt" ] || continue
        [ -f "${cupid_dir}/pose.json" ] || continue
        [ -f "${cupid_dir}/input_masked.png" ] || continue
        case_name="$(basename "$(dirname "$cupid_dir")")"
        count=$((count + 1))
        echo "--- [${count}] ${case_name} ---"
        run_case "$case_name"
        echo ""
    done
    if [ "$count" -eq 0 ]; then
        echo "No cases found under ${OUTPUT_DIR} (expected ${OUTPUT_DIR}/<case_name>/cupid/gaussians.pt)."
        exit 1
    fi
    echo "[done] ${count} case(s) processed."
fi
