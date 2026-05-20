#!/bin/bash
# Baseline stage 2: per-edge first-order optimization on top of CMA init.
#
# Usage:
#   bash scripts/baseline/train_warp.sh <case_name> [base_path]
#
# train_frame is read from <base_path>/<case_name>/split.json.

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/baseline/train_warp.sh <case_name> [base_path]"
    exit 1
fi

CASE_NAME="$1"
BASE_PATH="${2:-data/different_types}"

TRAIN_FRAME=$(python - <<PY
import json
print(json.load(open("${BASE_PATH}/${CASE_NAME}/split.json"))["train"][1])
PY
)

python train_warp.py \
    --base_path "${BASE_PATH}" \
    --case_name "${CASE_NAME}" \
    --train_frame "${TRAIN_FRAME}"
