#!/bin/bash
# Propagate features via symmetry (Step 4).
#
# Usage:
#   bash scripts/run_sym.sh <device_id>
#   bash scripts/run_sym.sh <device_id> [output_dir] [conda_env]
#   bash scripts/run_sym.sh <case_name> <device_id>
#   bash scripts/run_sym.sh <case_name> <device_id> [output_dir] [conda_env]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_sym.sh <device_id> [output_dir] [conda_env]"
    echo "   or: bash scripts/run_sym.sh <case_name> <device_id> [output_dir] [conda_env]"
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
        echo "Usage: bash scripts/run_sym.sh <case_name> <device_id> [output_dir] [conda_env]"
        exit 1
    fi
    CASE_NAME="$1"
    DEVICE_ID="$2"
    OUTPUT_DIR="${3:-results}"
    CONDA_ENV="${4:-phystwin}"
fi
MAX_MATCH_DIST="${MAX_MATCH_DIST:-0.1}"

run_case() {
    local case_name="$1"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "${CONDA_ENV}" \
        python semantic/gs_symmetry.py \
        --case_name "${case_name}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_match_dist "${MAX_MATCH_DIST}"
}

if [ -n "$CASE_NAME" ]; then
    echo "=== Step 4: Symmetry feature propagation ==="
    echo "  case:           ${CASE_NAME}"
    echo "  device_id:      ${DEVICE_ID}"
    echo "  output:         ${OUTPUT_DIR}"
    echo "  env:            ${CONDA_ENV}"
    echo "  max_match_dist: ${MAX_MATCH_DIST}"
    echo ""
    run_case "$CASE_NAME"
else
    echo "=== Step 4: Symmetry feature propagation (all cases) ==="
    echo "  device_id:      ${DEVICE_ID}"
    echo "  output:         ${OUTPUT_DIR}"
    echo "  env:            ${CONDA_ENV}"
    echo "  max_match_dist: ${MAX_MATCH_DIST}"
    echo ""
    count=0
    for vis_dir in "${OUTPUT_DIR}"/*/gaussian_vis; do
        [ -f "${vis_dir}/gaussians_vis.pt" ] || continue
        case_name="$(basename "$(dirname "$vis_dir")")"
        count=$((count + 1))
        echo "--- [${count}] ${case_name} ---"
        run_case "$case_name"
        echo ""
    done
    if [ "$count" -eq 0 ]; then
        echo "No cases found under ${OUTPUT_DIR} (expected ${OUTPUT_DIR}/<case_name>/gaussian_vis/gaussians_vis.pt)."
        exit 1
    fi
    echo "[done] ${count} case(s) processed."
fi
