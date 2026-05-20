#!/usr/bin/env bash
#
# Run Cupid 3D reconstruction.
#
# Usage:
#   bash scripts/run_cupid.sh <device_id>
#   bash scripts/run_cupid.sh <device_id> [base_path] [output_dir] [conda_env]
#   bash scripts/run_cupid.sh <case_name> <device_id>
#   bash scripts/run_cupid.sh <case_name> <device_id> [base_path] [output_dir] [conda_env]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_cupid.sh <device_id> [base_path] [output_dir] [conda_env]"
    echo "   or: bash scripts/run_cupid.sh <case_name> <device_id> [base_path] [output_dir] [conda_env]"
    exit 1
fi

CASE_NAME=""
if [[ "$1" =~ ^[0-9]+$ ]]; then
    DEVICE_ID="$1"
    BASE_PATH="${2:-data/different_types}"
    OUTPUT_DIR="${3:-results}"
    CUPID_ENV="${4:-${CUPID_ENV:-cupid}}"
else
    if [ $# -lt 2 ]; then
        echo "Single-case mode requires a device id."
        echo "Usage: bash scripts/run_cupid.sh <case_name> <device_id> [base_path] [output_dir] [conda_env]"
        exit 1
    fi
    CASE_NAME="$1"
    DEVICE_ID="$2"
    BASE_PATH="${3:-data/different_types}"
    OUTPUT_DIR="${4:-results}"
    CUPID_ENV="${5:-${CUPID_ENV:-cupid}}"
fi

run_single() {
    local case_name="$1"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "$CUPID_ENV" \
        python semantic/run_cupid_case.py \
        --case_name "$case_name" \
        --base_path "$BASE_PATH" \
        --output_dir "$OUTPUT_DIR"
}

if [ -n "$CASE_NAME" ]; then
    echo "[run_cupid] Running single case"
    echo "  case:      $CASE_NAME"
    echo "  device_id: $DEVICE_ID"
    echo "  base_path: $BASE_PATH"
    echo "  output:    $OUTPUT_DIR"
    echo "  env:       $CUPID_ENV"
    run_single "$CASE_NAME"
else
    echo "[run_cupid] Running all cases"
    echo "  device_id: $DEVICE_ID"
    echo "  base_path: $BASE_PATH"
    echo "  output:    $OUTPUT_DIR"
    echo "  env:       $CUPID_ENV"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" conda run -n "$CUPID_ENV" \
        python semantic/run_cupid_case.py \
        --base_path "$BASE_PATH" \
        --output_dir "$OUTPUT_DIR"
fi

echo "[run_cupid] Done."
