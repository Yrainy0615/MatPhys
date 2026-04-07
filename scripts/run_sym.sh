#!/bin/bash
# Propagate features via symmetry (Step 4).
#
# Usage:
#   bash scripts/run_sym.sh                    # all cases under output_dir
#   bash scripts/run_sym.sh <case_name>        # single case
#   bash scripts/run_sym.sh <case_name> [output_dir] [conda_env]
#
# Examples:
#   bash scripts/run_sym.sh
#   bash scripts/run_sym.sh single_lift_cloth_4
#   bash scripts/run_sym.sh single_lift_cloth_4 results phystwin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${2:-results}"
CONDA_ENV="${3:-phystwin}"
MAX_MATCH_DIST="${MAX_MATCH_DIST:-0.1}"

if [ $# -ge 1 ] && [ -n "$1" ]; then
    # Single case
    CASE_NAME="$1"
    echo "=== Step 4: Symmetry feature propagation ==="
    echo "  case:           ${CASE_NAME}"
    echo "  output:         ${OUTPUT_DIR}"
    echo "  env:            ${CONDA_ENV}"
    echo "  max_match_dist: ${MAX_MATCH_DIST}"
    echo ""
    conda run -n "${CONDA_ENV}" python semantic/gs_symmetry.py \
        --case_name "${CASE_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_match_dist "${MAX_MATCH_DIST}"
else
    # All cases: discover from output_dir (any dir with gaussian_vis/gaussians_vis.pt)
    echo "=== Step 4: Symmetry feature propagation (all cases) ==="
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
        conda run -n "${CONDA_ENV}" python semantic/gs_symmetry.py \
            --case_name "${case_name}" \
            --output_dir "${OUTPUT_DIR}" \
            --max_match_dist "${MAX_MATCH_DIST}"
        echo ""
    done
    if [ "$count" -eq 0 ]; then
        echo "No cases found under ${OUTPUT_DIR} (expected ${OUTPUT_DIR}/<case_name>/gaussian_vis/gaussians_vis.pt)."
        exit 1
    fi
    echo "[done] ${count} case(s) processed."
fi
