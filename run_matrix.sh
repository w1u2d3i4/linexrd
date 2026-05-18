#!/bin/bash
# LineXRD autoresearch matrix: 4 backbone × ccdc_sg × 12 epoch baseline
# 后台串行执行，把日志写到 logs/

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"     # reduced from 128 to fit on shared A100-40G
D_MODEL="${D_MODEL:-256}"
DEPTH="${DEPTH:-6}"                # reduced from 8 to fit memory for retnet
DATASET="${DATASET:-ccdc_sg}"

mkdir -p logs results

# Note: mamba 因为是纯 PyTorch 循环 scan, L=4251 会非常慢,
# 在 smoke matrix 里暂时跑 mamba 仅 1 epoch 验证可跑通
for BACKBONE in cnn gla retnet; do
    echo "[matrix] starting BACKBONE=$BACKBONE"
    BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE \
        D_MODEL=$D_MODEL DEPTH=$DEPTH bash autoresearch.sh \
        > logs/matrix_${BACKBONE}_${DATASET}.out 2>&1
    echo "[matrix] done BACKBONE=$BACKBONE  --  $(date)"
done

echo "[matrix] ALL DONE"
