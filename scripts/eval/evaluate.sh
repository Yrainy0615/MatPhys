#!/bin/bash
cd "$(dirname "$0")/../.."
set -e

CASE_NAME=""
if [ "$#" -ge 2 ] && [ "$1" = "--case_name" ]; then
  CASE_NAME="$2"
fi

if [ -n "$CASE_NAME" ]; then
  python evaluate_chamfer.py --case_name "$CASE_NAME"
  python evaluate_track.py --case_name "$CASE_NAME"
  python gaussian_splatting/evaluate_render.py --case_name "$CASE_NAME"
else
  python evaluate_chamfer.py
  python evaluate_track.py
  python gaussian_splatting/evaluate_render.py
fi
