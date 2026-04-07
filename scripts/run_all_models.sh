#!/bin/bash
# Run all four model types with physics simulation training
# Usage: bash scripts/run_all_models.sh

set -e

# Configuration
BASE_DIR="checkpoints/model_comparison"
EPOCHS=1000
BATCH_SIZE=4
LR=0.001
SEED=42
TRAIN_RATIO=0.8
EVAL_EVERY=10
DEVICE="cuda"

# Loss weights
LAMBDA_TRACK=1.0
LAMBDA_GEO=1.0
LAMBDA_RENDER=1.0

# Model types to test
MODEL_TYPES=("edge_level" "part_level" "hybrid" "pure_dino")

# Create base directory
mkdir -p "$BASE_DIR"

echo "============================================================"
echo "Running All Model Types with Physics Simulation"
echo "============================================================"
echo "Base directory: $BASE_DIR"
echo "Epochs: $EPOCHS"
echo "Train/Test split: ${TRAIN_RATIO}:$(echo "1 - $TRAIN_RATIO" | bc)"
echo "Loss weights: track=$LAMBDA_TRACK, geo=$LAMBDA_GEO, render=$LAMBDA_RENDER"
echo "Models: ${MODEL_TYPES[*]}"
echo "============================================================"
echo ""

# Run training for each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    SAVE_DIR="${BASE_DIR}/${MODEL_TYPE}"
    mkdir -p "$SAVE_DIR"

    echo "============================================================"
    echo "Training: $MODEL_TYPE"
    echo "Save directory: $SAVE_DIR"
    echo "============================================================"

    python semantic/train_models.py \
        --model_type "$MODEL_TYPE" \
        --save_dir "$SAVE_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        --train_ratio "$TRAIN_RATIO" \
        --eval_every "$EVAL_EVERY" \
        --lambda_track "$LAMBDA_TRACK" \
        --lambda_geo "$LAMBDA_GEO" \
        --lambda_render "$LAMBDA_RENDER" \
        --device "$DEVICE"

    echo ""
    echo "Completed: $MODEL_TYPE"
    echo ""
done

echo "============================================================"
echo "All Training Complete!"
echo "============================================================"
echo ""

# Compare results
echo "============================================================"
echo "Results Comparison (Test Set)"
echo "============================================================"
printf "%-15s | %-12s | %-12s | %-12s | %-12s\n" "Model Type" "Total Loss" "Track Loss" "Geo Loss" "Render Loss"
echo "---------------------------------------------------------------------------------"

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    METRICS_FILE="${BASE_DIR}/${MODEL_TYPE}/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        TOTAL=$(python -c "import json; d=json.load(open('$METRICS_FILE')); print(f\"{d['final_metrics']['total']:.6f}\")")
        TRACK=$(python -c "import json; d=json.load(open('$METRICS_FILE')); print(f\"{d['final_metrics']['track']:.6f}\")")
        GEO=$(python -c "import json; d=json.load(open('$METRICS_FILE')); print(f\"{d['final_metrics']['geo']:.6f}\")")
        RENDER=$(python -c "import json; d=json.load(open('$METRICS_FILE')); print(f\"{d['final_metrics']['render']:.6f}\")")
        printf "%-15s | %-12s | %-12s | %-12s | %-12s\n" "$MODEL_TYPE" "$TOTAL" "$TRACK" "$GEO" "$RENDER"
    else
        printf "%-15s | %-12s | %-12s | %-12s | %-12s\n" "$MODEL_TYPE" "N/A" "N/A" "N/A" "N/A"
    fi
done

echo "============================================================"
echo ""

# Find best model
echo "Finding best model based on Total Test Loss..."
BEST_MODEL=""
BEST_LOSS="999999"

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    METRICS_FILE="${BASE_DIR}/${MODEL_TYPE}/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        TOTAL=$(python -c "import json; d=json.load(open('$METRICS_FILE')); print(d['final_metrics']['total'])")
        IS_BETTER=$(python -c "print(1 if $TOTAL < $BEST_LOSS else 0)")
        if [ "$IS_BETTER" -eq 1 ]; then
            BEST_LOSS=$TOTAL
            BEST_MODEL=$MODEL_TYPE
        fi
    fi
done

echo "Best Model: $BEST_MODEL (Total Loss: $BEST_LOSS)"
echo ""
echo "Results saved to: $BASE_DIR"
