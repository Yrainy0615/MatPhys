#!/bin/bash
# Extract GPT-4o physics priors for all cases.
# Saves results/<case>/gpt_physics_prior.json for each case.
#
# Usage:
#   bash scripts/run_extract_gpt_prior.sh <OPENAI_API_KEY>
#   or: export OPENAI_API_KEY=sk-...; bash scripts/run_extract_gpt_prior.sh

set -euo pipefail
cd "$(dirname "$0")/../../.."

API_KEY="${1:-${OPENAI_API_KEY:-}}"
if [ -z "$API_KEY" ]; then
    echo "ERROR: provide API key as first argument or set OPENAI_API_KEY"
    exit 1
fi

CASES=(
    double_lift_cloth_1
    double_lift_cloth_3
    double_lift_sloth
    double_lift_zebra
    double_stretch_sloth
    double_stretch_zebra
    rope_double_hand
    single_clift_cloth_1
    single_clift_cloth_3
    single_lift_cloth
    single_lift_cloth_1
    single_lift_cloth_3
    single_lift_cloth_4
    single_lift_dinosor
    single_lift_rope
    single_lift_sloth
    single_lift_zebra
    single_push_rope
    single_push_rope_1
    single_push_rope_4
    single_push_sloth
    weird_package
)

echo "[$(date)] Extracting GPT physics priors for ${#CASES[@]} cases..."

python semantic/query_physics_prior_gpt.py \
    --base_path data/different_types \
    --results_dir results \
    --cases "${CASES[@]}" \
    --api_key "$API_KEY" \
    --gpt_model gpt-4o \
    --skip_existing

echo "[$(date)] Done. Check results/<case>/gpt_physics_prior.json"
