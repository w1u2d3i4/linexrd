#!/bin/bash
set -euo pipefail

# ============================================================
# LineXRD Autoresearch Benchmark Script
# 输出 METRIC name=value 行供 runner.py 解析
# ============================================================

PYTHON="/opt/data/private/icml2026/bin/python"   # user-requested env
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------- HYPERPARAMS (agent may modify) -------
BACKBONE="${BACKBONE:-cnn}"        # cnn / gla / mamba / retnet
DATASET="${DATASET:-ccdc_sg}"      # ccdc_sg / inor / coremof19
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
D_MODEL="${D_MODEL:-256}"
DEPTH="${DEPTH:-8}"
USE_HIER="${USE_HIER:-0}"          # 0 / 1
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
SEED="${SEED:-0}"
DOWNSAMPLE_TO="${DOWNSAMPLE_TO:-0}"   # 0 = full resolution, >0 = e.g. 2000
# ------- END HYPERPARAMS -------

EXPERIMENT_NAME="${BACKBONE}_${DATASET}_d${D_MODEL}_dp${DEPTH}_ep${EPOCHS}_s${SEED}"
SAVE_DIR="${SRC_DIR}/checkpoints/${EXPERIMENT_NAME}"
LOG_FILE="${SRC_DIR}/logs/${EXPERIMENT_NAME}.log"
mkdir -p "$SAVE_DIR" "${SRC_DIR}/logs" "${SRC_DIR}/results"

echo "=== TRAINING: ${EXPERIMENT_NAME} ==="

CMD=("$PYTHON" -m src.train_linexrd \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --d_model "$D_MODEL" \
    --depth "$DEPTH" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED")
if [ "$USE_HIER" = "1" ]; then CMD+=(--use_hier_head); fi
if [ -n "$MAX_TRAIN_BATCHES" ]; then CMD+=(--max_train_batches "$MAX_TRAIN_BATCHES"); fi
if [ "$DOWNSAMPLE_TO" != "0" ]; then CMD+=(--downsample_to "$DOWNSAMPLE_TO"); fi

cd "$SRC_DIR"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "=== DONE: ${EXPERIMENT_NAME} ==="
