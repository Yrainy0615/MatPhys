#!/bin/bash
# Extract material distribution and part segmentation (Step 5).
#
# Usage:
#   bash scripts/run_material.sh                    # all cases under output_dir
#   bash scripts/run_material.sh <case_name>        # single case
#   bash scripts/run_material.sh <case_name> [output_dir] [conda_env]
#
# Environment variables:
#   SAM_CHECKPOINT    Path to SAM checkpoint (default: sam_vit_h_4b8939.pth)
#   SKIP_SAM=1        Skip SAM, use single-part fallback
#   SKIP_CLIP=1       Skip CLIP, use uniform material distribution
#
# Examples:
#   bash scripts/run_material.sh
#   bash scripts/run_material.sh single_lift_cloth_4
#   SKIP_SAM=1 bash scripts/run_material.sh single_lift_cloth_4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${2:-results}"
CONDA_ENV="${3:-phystwin}"

# Build extra args from env vars
EXTRA_ARGS=""
if [ "${SKIP_SAM:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_sam"
fi
if [ "${SKIP_CLIP:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_clip"
fi

if [ $# -ge 1 ] && [ -n "$1" ]; then
    # Single case
    CASE_NAME="$1"
    echo "=== Step 5: Material distribution and part segmentation ==="
    echo "  case:    ${CASE_NAME}"
    echo "  output:  ${OUTPUT_DIR}"
    echo "  env:     ${CONDA_ENV}"
    [ -n "$EXTRA_ARGS" ] && echo "  extra:   ${EXTRA_ARGS}"
    echo ""
    conda run -n "${CONDA_ENV}" python semantic/extract_material_parts.py \
        --case_name "${CASE_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        $EXTRA_ARGS
else
    # All cases: discover from output_dir (any dir with gaussian_vis/)
    echo "=== Step 5: Material distribution and part segmentation (all cases) ==="
    echo "  output:  ${OUTPUT_DIR}"
    echo "  env:     ${CONDA_ENV}"
    [ -n "$EXTRA_ARGS" ] && echo "  extra:   ${EXTRA_ARGS}"
    echo ""
    count=0
    for vis_dir in "${OUTPUT_DIR}"/*/gaussian_vis; do
        # Need either gaussians_vis_sym.pt or gaussians_vis.pt
        if [ ! -f "${vis_dir}/gaussians_vis_sym.pt" ] && [ ! -f "${vis_dir}/gaussians_vis.pt" ]; then
            continue
        fi
        case_name="$(basename "$(dirname "$vis_dir")")"
        count=$((count + 1))
        echo "--- [${count}] ${case_name} ---"
        conda run -n "${CONDA_ENV}" python semantic/extract_material_parts.py \
            --case_name "${case_name}" \
            --output_dir "${OUTPUT_DIR}" \
            $EXTRA_ARGS
        echo ""
    done
    if [ "$count" -eq 0 ]; then
        echo "No cases found under ${OUTPUT_DIR} (expected ${OUTPUT_DIR}/<case_name>/gaussian_vis/gaussians_vis*.pt)."
        exit 1
    fi
    echo "[done] ${count} case(s) processed."
fi
