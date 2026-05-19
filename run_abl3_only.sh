#!/bin/bash
# Ablation 3 only: cross-dataset transfer to INOR / CoREMOF19, 40 epoch,
# full resolution. Standalone driver so we don't re-run abl1 (already done).

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

EPOCHS=40
BATCH_SIZE=64
D_MODEL=256
DEPTH=6
LR=2e-4
DOWNSAMPLE_TO=0

for DATASET in inor coremof19; do
    for BACKBONE in cnn gla retnet; do
        echo "[abl3-$DATASET] start BACKBONE=$BACKBONE  -- $(date)" \
            | tee -a logs/ablations_master.log
        BACKBONE=$BACKBONE DATASET=$DATASET EPOCHS=$EPOCHS \
            BATCH_SIZE=$BATCH_SIZE D_MODEL=$D_MODEL DEPTH=$DEPTH LR=$LR \
            DOWNSAMPLE_TO=$DOWNSAMPLE_TO \
            bash train.sh > logs/abl3_${BACKBONE}_${DATASET}.out 2>&1
        echo "[abl3-$DATASET] done BACKBONE=$BACKBONE  -- $(date)" \
            | tee -a logs/ablations_master.log
    done
done

echo "[abl3-only] ALL DONE  -- $(date)" | tee -a logs/ablations_master.log
