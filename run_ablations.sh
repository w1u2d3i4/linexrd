#!/bin/bash
# Ablation 1 (downsample 8500->2000) + Ablation 3 (cross-dataset transfer)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs

# ===== Ablation 1: downsample input on CCDC-SG, 20 epoch =====
EPOCHS=20 BATCH_SIZE=64 D_MODEL=256 DEPTH=6 LR=2e-4
DOWNSAMPLE_TO=2000 DATASET=ccdc_sg
for BACKBONE in cnn gla retnet; do
    echo "[abl1-ds2000] start BACKBONE=$BACKBONE  -- $(date)"
    BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE \
        D_MODEL=$D_MODEL DEPTH=$DEPTH LR=$LR DOWNSAMPLE_TO=$DOWNSAMPLE_TO \
        bash autoresearch.sh > logs/abl1_${BACKBONE}_ds2000.out 2>&1
    echo "[abl1-ds2000] done BACKBONE=$BACKBONE  -- $(date)"
done

# ===== Ablation 3: cross-dataset transfer, full resolution, 40 epoch =====
EPOCHS=40 DOWNSAMPLE_TO=0
for DATASET in inor coremof19; do
    for BACKBONE in cnn gla retnet; do
        echo "[abl3-$DATASET] start BACKBONE=$BACKBONE  -- $(date)"
        BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE \
            D_MODEL=$D_MODEL DEPTH=$DEPTH LR=$LR DOWNSAMPLE_TO=$DOWNSAMPLE_TO \
            bash autoresearch.sh > logs/abl3_${BACKBONE}_${DATASET}.out 2>&1
        echo "[abl3-$DATASET] done BACKBONE=$BACKBONE  -- $(date)"
    done
done

echo "[ablations] ALL DONE  -- $(date)"
