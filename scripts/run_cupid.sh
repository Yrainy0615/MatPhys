#!/usr/bin/env bash
#
# Run Cupid 3D reconstruction for all cases.
#
# Usage:
#   bash scripts/run_cupid.sh                         # all cases
#   bash scripts/run_cupid.sh single_lift_cloth       # single case
#   CUPID_ENV=cupid3d bash scripts/run_cupid.sh       # custom env name
#
# The script uses `conda run` so it does not require `conda activate`.
# Set CUPID_ENV to change the conda environment name (default: cupid).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUPID_ENV="${CUPID_ENV:-cupid}"
BASE_PATH="${BASE_PATH:-data/different_types}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

cd "$PROJECT_ROOT"

if [ $# -ge 1 ]; then
    echo "[run_cupid] Running single case: $1"
    conda run -n "$CUPID_ENV" \
        python semantic/run_cupid_case.py \
            --case_name "$1" \
            --base_path "$BASE_PATH" \
            --output_dir "$OUTPUT_DIR"
else
    echo "[run_cupid] Running ALL cases under $BASE_PATH"
    conda run -n "$CUPID_ENV" \
        python semantic/run_cupid_case.py \
            --base_path "$BASE_PATH" \
            --output_dir "$OUTPUT_DIR"
fi

echo "[run_cupid] Done."
