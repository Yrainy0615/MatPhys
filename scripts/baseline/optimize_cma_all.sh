#!/bin/bash
# Baseline stage 1 over all cases under <base_path>.
# Replaces the old script_optimize.py launcher.
#
# Usage:
#   bash scripts/baseline/optimize_cma_all.sh [base_path] [max_iter]

set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_PATH="${1:-data/different_types}"
MAX_ITER="${2:-20}"

for case_dir in "${BASE_PATH}"/*/; do
    CASE_NAME="$(basename "$case_dir")"
    [ -f "${case_dir}/split.json" ] || { echo "[skip] ${CASE_NAME}: no split.json"; continue; }
    echo "=== optimize_cma: ${CASE_NAME} ==="
    bash scripts/baseline/optimize_cma.sh "${CASE_NAME}" "${BASE_PATH}" "${MAX_ITER}"
done
