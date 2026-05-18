#!/bin/bash
# LineXRD canonical 3-backbone matrix on CCDC-SG, 40 epoch, single seed,
# class-weighted cross-entropy (sqrt-inverse class frequencies).
#
# Reproduces the headline result table in the paper.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EPOCHS=40
BATCH_SIZE=64
D_MODEL=256
DEPTH=6
LR=2e-4
DATASET=ccdc_sg

for BACKBONE in cnn gla retnet; do
    echo "[matrix] starting BACKBONE=$BACKBONE  -- $(date)"
    BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS \
        BATCH_SIZE=$BATCH_SIZE D_MODEL=$D_MODEL DEPTH=$DEPTH LR=$LR \
        bash train.sh
    echo "[matrix] done BACKBONE=$BACKBONE  -- $(date)"
done

echo "[matrix] ALL DONE  -- $(date)"
