#!/bin/bash
set -euo pipefail

# ============================================================
# LineXRD training entry point.
#
# All hyperparameters are exposed as environment variables so the same
# script can be invoked by run_matrix.sh / run_ablations.sh as well as
# directly from the command line.
#
# Usage:
#   bash train.sh                                      # defaults
#   BACKBONE=gla EPOCHS=40 bash train.sh               # override one knob
#   DATASET=inor BACKBONE=retnet LR=2e-4 bash train.sh # cross-dataset
# ============================================================

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------- hyperparameters (overridable from env) -------
BACKBONE="${BACKBONE:-cnn}"            # cnn / gla / mamba / retnet
DATASET="${DATASET:-ccdc_sg}"          # ccdc_sg / inor / coremof19
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
D_MODEL="${D_MODEL:-256}"
DEPTH="${DEPTH:-6}"
USE_HIER="${USE_HIER:-0}"              # 0 / 1
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
SEED="${SEED:-0}"
DOWNSAMPLE_TO="${DOWNSAMPLE_TO:-0}"    # 0 = full resolution, >0 = e.g. 2000
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
CLASS_WEIGHT="${CLASS_WEIGHT:-sqrt_inv}"   # none / sqrt_inv / inv
# -------------------------------------------------------

RUN="${BACKBONE}_${DATASET}_d${D_MODEL}_dp${DEPTH}_ep${EPOCHS}_s${SEED}"
[ "$DOWNSAMPLE_TO" != "0" ] && RUN="${RUN}_ds${DOWNSAMPLE_TO}"
SAVE_DIR="${ROOT}/checkpoints/${RUN}"
LOG_FILE="${ROOT}/logs/${RUN}.log"
mkdir -p "$SAVE_DIR" "${ROOT}/logs"

echo "=== TRAINING: ${RUN} ==="

CMD=("$PYTHON" -m src.train_linexrd
    --backbone "$BACKBONE"
    --dataset "$DATASET"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --d_model "$D_MODEL"
    --depth "$DEPTH"
    --save_dir "$SAVE_DIR"
    --seed "$SEED"
    --label_smoothing "$LABEL_SMOOTHING"
    --class_weight "$CLASS_WEIGHT")
[ "$USE_HIER" = "1" ]               && CMD+=(--use_hier_head)
[ -n "$MAX_TRAIN_BATCHES" ]         && CMD+=(--max_train_batches "$MAX_TRAIN_BATCHES")
[ "$DOWNSAMPLE_TO" != "0" ]         && CMD+=(--downsample_to "$DOWNSAMPLE_TO")

cd "$ROOT"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "=== DONE: ${RUN} ==="
