#!/bin/bash
# Multi-view PhysTwin preprocessing over every case in data_config.csv.
# Replaces script_process_data.py.
#
# Usage:
#   bash scripts/preprocess/multi_view_all.sh [base_path] [data_config_csv]

set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_PATH="${1:-data/different_types}"
DATA_CSV="${2:-data_config.csv}"

rm -f timer.log

while IFS=, read -r CASE_NAME CATEGORY SHAPE_PRIOR _; do
    [ -d "${BASE_PATH}/${CASE_NAME}" ] || continue
    echo "=== process_data: ${CASE_NAME} (${CATEGORY}) ==="
    if [ "$(echo "${SHAPE_PRIOR}" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
        python process_data.py --base_path "${BASE_PATH}" --case_name "${CASE_NAME}" --category "${CATEGORY}" --shape_prior
    else
        python process_data.py --base_path "${BASE_PATH}" --case_name "${CASE_NAME}" --category "${CATEGORY}"
    fi
done < "${DATA_CSV}"
