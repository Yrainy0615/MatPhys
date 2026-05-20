#!/bin/bash
# Extract material distribution and part segmentation (Step 5).
#
# Usage:
#   bash scripts/run_material.sh <device_id>
#   bash scripts/run_material.sh <device_id> [output_dir] [conda_env]
#   bash scripts/run_material.sh <case_name> <device_id>
#   bash scripts/run_material.sh <case_name> <device_id> [output_dir] [conda_env]
#
# Environment variables:
#   SAM_CHECKPOINT    Path to SAM checkpoint (default: sam_vit_h_4b8939.pth)
#   SKIP_SAM=1        Skip SAM, use single-part fallback
#   SKIP_CLIP=1       Skip CLIP, use uniform material distribution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_material.sh <device_id> [output_dir] [conda_env]"
    echo "   or: bash scripts/run_material.sh <case_name> <device_id> [output_dir] [conda_env]"
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
        echo "Usage: bash scripts/run_material.sh <case_name> <device_id> [output_dir] [conda_env]"
        exit 1
    fi
    CASE_NAME="$1"
    DEVICE_ID="$2"
    OUTPUT_DIR="${3:-results}"
    CONDA_ENV="${4:-phystwin}"
fi

EXTRA_ARGS=""
if [ "${SKIP_SAM:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_sam"
fi
if [ "${SKIP_CLIP:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_clip"
fi

run_case() {
    local case_name="$1"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${CONDA_ENV}" \
        python semantic/extract_material_parts.py \
        --case_name "${case_name}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda \
        ${EXTRA_ARGS}
}

if [ -n "$CASE_NAME" ]; then
    echo "=== Step 5: Material distribution and part segmentation ==="
    echo "  case:      ${CASE_NAME}"
    echo "  device_id: ${DEVICE_ID}"
    echo "  output:    ${OUTPUT_DIR}"
    echo "  env:       ${CONDA_ENV}"
    [ -n "$EXTRA_ARGS" ] && echo "  extra:     ${EXTRA_ARGS}"
    echo ""
    run_case "$CASE_NAME"
else
    echo "=== Step 5: Material distribution and part segmentation (all cases) ==="
    echo "  device_id: ${DEVICE_ID}"
    echo "  output:    ${OUTPUT_DIR}"
    echo "  env:       ${CONDA_ENV}"
    [ -n "$EXTRA_ARGS" ] && echo "  extra:     ${EXTRA_ARGS}"
    echo ""
    count=0
    for vis_dir in "${OUTPUT_DIR}"/*/gaussian_vis; do
        if [ ! -f "${vis_dir}/gaussians_vis_sym.pt" ] && [ ! -f "${vis_dir}/gaussians_vis.pt" ]; then
            continue
        fi
        case_name="$(basename "$(dirname "$vis_dir")")"
        count=$((count + 1))
        echo "--- [${count}] ${case_name} ---"
        run_case "$case_name"
        echo ""
    done
    if [ "$count" -eq 0 ]; then
        echo "No cases found under ${OUTPUT_DIR} (expected ${OUTPUT_DIR}/<case_name>/gaussian_vis/gaussians_vis*.pt)."
        exit 1
    fi
    echo "[done] ${count} case(s) processed."
fi
