#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=CLIP_ADAPTER

DATASET=$1
SEED=$2

CFG=lr_1e-3_layers_4_epochs_5_base2new
SHOTS=16

BATCH_SIZE=$3
LR=$4
EPOCHS=$5
GPU=$6


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${EPOCHS}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISISBLE_DEVICES=$GPU python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr        ${LR} \
    --ep        ${EPOCHS} \
    --trainer   ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISISBLE_DEVICES=$GPU python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr       ${LR} \
    --ep        ${EPOCHS} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi