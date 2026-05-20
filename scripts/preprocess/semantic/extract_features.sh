#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_semantic_extraction.sh <base_path> <device_id> [output_dir]"
    echo "Example: bash scripts/run_semantic_extraction.sh data/different_types 2"
    exit 1
fi

BASE_PATH="$1"
DEVICE_ID="$2"
OUTPUT_DIR="${3:-semantic/cache}"
CASE_MAP_PATH="semantic/case_to_material_different_types.json"

if [ ! -d "$BASE_PATH" ]; then
    echo "Base path not found: $BASE_PATH"
    exit 1
fi

if [ ! -f "$CASE_MAP_PATH" ]; then
    echo "Case mapping not found: $CASE_MAP_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CASES=$(python - <<'PY'
import json
from pathlib import Path

case_map_path = Path("semantic/case_to_material_different_types.json")
data = json.load(case_map_path.open())
print(" ".join(sorted(data["case_to_material"].keys())))
PY
)

if [ -z "$CASES" ]; then
    echo "No cases found in $CASE_MAP_PATH"
    exit 1
fi

echo "Running semantic feature extraction"
echo "  base_path: $BASE_PATH"
echo "  device_id: $DEVICE_ID"
echo "  output_dir: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$DEVICE_ID" python semantic/extract_dino_semantic_features.py \
    --base_path "$BASE_PATH" \
    --cases $CASES \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --frame_idx 0
