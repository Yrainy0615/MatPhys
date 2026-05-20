#!/bin/bash
# Baseline inference over all cases that have a trained checkpoint under
# experiments/<case_name>. Replaces script_inference.py.
#
# Usage:
#   bash scripts/baseline/inference_warp_all.sh [base_path] [experiments_dir]

set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_PATH="${1:-data/different_types}"
EXP_DIR="${2:-experiments}"

for case_dir in "${EXP_DIR}"/*/; do
    CASE_NAME="$(basename "$case_dir")"
    echo "=== inference_warp: ${CASE_NAME} ==="
    bash scripts/baseline/inference_warp.sh "${CASE_NAME}" "${BASE_PATH}"
done
