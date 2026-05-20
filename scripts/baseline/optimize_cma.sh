#!/bin/bash
# Baseline stage 1: CMA-ES global optimization of sparse spring params.
#
# Usage:
#   bash scripts/baseline/optimize_cma.sh <case_name> [base_path] [max_iter]
#
# train_frame is read from <base_path>/<case_name>/split.json.

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/baseline/optimize_cma.sh <case_name> [base_path] [max_iter]"
    exit 1
fi

CASE_NAME="$1"
BASE_PATH="${2:-data/different_types}"
MAX_ITER="${3:-20}"

TRAIN_FRAME=$(python - <<PY
import json
print(json.load(open("${BASE_PATH}/${CASE_NAME}/split.json"))["train"][1])
PY
)

python optimize_cma.py \
    --base_path "${BASE_PATH}" \
    --case_name "${CASE_NAME}" \
    --train_frame "${TRAIN_FRAME}" \
    --max_iter "${MAX_ITER}"
