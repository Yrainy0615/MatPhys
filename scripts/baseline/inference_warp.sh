#!/bin/bash
# Baseline inference: roll out optimized warp params on one case.
#
# Usage:
#   bash scripts/baseline/inference_warp.sh <case_name> [base_path]

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/baseline/inference_warp.sh <case_name> [base_path]"
    exit 1
fi

CASE_NAME="$1"
BASE_PATH="${2:-data/different_types}"

python inference_warp.py \
    --base_path "${BASE_PATH}" \
    --case_name "${CASE_NAME}"
