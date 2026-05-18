#!/bin/bash
# LineXRD matrix v3 (post-fix):
#   * 3 backbones (CNN, GLA-fixed, RetNet-chunkwise) on CCDC-SG
#   * 40 epochs, batch 64, LR 5e-4, d_model=256, depth=6, 1 seed
#   * GPU 1 only (GPU 0 has another tenant)

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=1

EPOCHS=40
BATCH_SIZE=64
D_MODEL=256
DEPTH=6
DATASET=ccdc_sg
LR=5e-4

mkdir -p logs

for BACKBONE in cnn gla retnet; do
    echo "[matrix_v3] starting BACKBONE=$BACKBONE  --  $(date)"
    BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE \
        D_MODEL=$D_MODEL DEPTH=$DEPTH LR=$LR bash autoresearch.sh \
        > logs/v3_${BACKBONE}_${DATASET}.out 2>&1
    echo "[matrix_v3] done BACKBONE=$BACKBONE  --  $(date)"
done

echo "[matrix_v3] ALL DONE  --  $(date)"
