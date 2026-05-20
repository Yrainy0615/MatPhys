#!/bin/bash
# Baseline stage 2 over all cases. Replaces script_train.py.
#
# Usage:
#   bash scripts/baseline/train_warp_all.sh [base_path]

set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_PATH="${1:-data/different_types}"

for case_dir in "${BASE_PATH}"/*/; do
    CASE_NAME="$(basename "$case_dir")"
    [ -f "${case_dir}/split.json" ] || { echo "[skip] ${CASE_NAME}: no split.json"; continue; }
    echo "=== train_warp: ${CASE_NAME} ==="
    bash scripts/baseline/train_warp.sh "${CASE_NAME}" "${BASE_PATH}"
done
