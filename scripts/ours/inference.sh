#!/bin/bash
# Ours inference: roll out a trained per-case predictor checkpoint and
# evaluate (track + chamfer) on the chosen split. Optionally save
# per-case visualization videos.
#
# Usage:
#   bash scripts/ours/inference.sh <ckpt> [device] [split] [cases]
#
# Args:
#   ckpt    path to a .pth (e.g. checkpoints/per_case_smooth_fitall_*/<case>/last_checkpoint.pth)
#   device  CUDA device, default cuda:0
#   split   one of {test, train, both}, default test
#   cases   comma-separated case names to restrict eval to (optional)
#
# Outputs:
#   <ckpt>.eval.json                metrics
#   <ckpt_dir>/eval_vis/<case>.mp4  visualization videos (if VIS=1)

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/ours/inference.sh <ckpt> [device] [split] [cases]"
    exit 1
fi

CKPT="$1"
DEVICE="${2:-cuda:0}"
SPLIT="${3:-test}"
CASES="${4:-}"

CKPT_DIR="$(dirname "${CKPT}")"
OUT_JSON="${CKPT%.pth}.eval.json"

EXTRA=()
if [ -n "${CASES}" ]; then EXTRA+=(--cases "${CASES}"); fi
if [ "${VIS:-0}" = "1" ]; then
    VIS_DIR="${CKPT_DIR}/eval_vis"
    mkdir -p "${VIS_DIR}"
    EXTRA+=(--vis_dir "${VIS_DIR}")
fi

python semantic/eval_simple_video.py \
    --ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --split "${SPLIT}" \
    --out_json "${OUT_JSON}" \
    "${EXTRA[@]}"

echo "[done] metrics: ${OUT_JSON}"
