#!/bin/bash
# Lift DINOv2 features to visible Gaussians (Step 2).
#
# Usage:
#   bash scripts/run_lift_dino.sh                    # all cases under output_dir
#   bash scripts/run_lift_dino.sh <case_name>        # single case
#   bash scripts/run_lift_dino.sh <case_name> [output_dir] [conda_env]
#
# Examples:
#   bash scripts/run_lift_dino.sh
#   bash scripts/run_lift_dino.sh single_lift_cloth_4
#   bash scripts/run_lift_dino.sh single_lift_cloth_4 results phystwin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${2:-results}"
CONDA_ENV="${3:-phystwin}"

if [ $# -ge 1 ] && [ -n "$1" ]; then
    # Single case
    CASE_NAME="$1"
    echo "=== Step 2: Lift DINO features to Gaussians ==="
    echo "  case:    ${CASE_NAME}"
    echo "  output:  ${OUTPUT_DIR}"
    echo "  env:     ${CONDA_ENV}"
    echo ""
    conda run -n "${CONDA_ENV}" python semantic/lift_dino_to_gaussian.py \
        --case_name "${CASE_NAME}" \
        --output_dir "${OUTPUT_DIR}"
else
    # All cases: discover from output_dir (any dir with cupid/gaussians.pt)
    echo "=== Step 2: Lift DINO features to Gaussians (all cases) ==="
    echo "  output:  ${OUTPUT_DIR}"
    echo "  env:     ${CONDA_ENV}"
    echo ""
    count=0
    for cupid_dir in "${OUTPUT_DIR}"/*/cupid; do
        [ -f "${cupid_dir}/gaussians.pt" ] || continue
        [ -f "${cupid_dir}/pose.json" ] || continue
        [ -f "${cupid_dir}/input_masked.png" ] || continue
        case_name="$(basename "$(dirname "$cupid_dir")")"
        count=$((count + 1))
        echo "--- [${count}] ${case_name} ---"
        conda run -n "${CONDA_ENV}" python semantic/lift_dino_to_gaussian.py \
            --case_name "${case_name}" \
            --output_dir "${OUTPUT_DIR}"
        echo ""
    done
    if [ "$count" -eq 0 ]; then
        echo "No cases found under ${OUTPUT_DIR} (expected ${OUTPUT_DIR}/<case_name>/cupid/gaussians.pt)."
        exit 1
    fi
    echo "[done] ${count} case(s) processed."
fi
