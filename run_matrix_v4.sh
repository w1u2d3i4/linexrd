#!/bin/bash
# LineXRD matrix v4 — post class-imbalance fix
#   * class_weight = sqrt_inv (combats 22.45% P-1 majority-class collapse)
#   * label_smoothing = 0.1
#   * LR 2e-4 (down from 5e-4 to stabilise late epochs)
#   * 3 backbones (CNN, GLA-fixed, RetNet-chunkwise) on CCDC-SG
#   * 40 epochs, batch 64, d_model=256, depth=6, 1 seed
#   * GPU 1 only

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=1

EPOCHS=40
BATCH_SIZE=64
D_MODEL=256
DEPTH=6
DATASET=ccdc_sg
LR=2e-4

mkdir -p logs

for BACKBONE in cnn gla retnet; do
    echo "[matrix_v4] starting BACKBONE=$BACKBONE  --  $(date)"
    BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE \
        D_MODEL=$D_MODEL DEPTH=$DEPTH LR=$LR bash autoresearch.sh \
        > logs/v4_${BACKBONE}_${DATASET}.out 2>&1
    echo "[matrix_v4] done BACKBONE=$BACKBONE  --  $(date)"
done

echo "[matrix_v4] ALL DONE  --  $(date)"
