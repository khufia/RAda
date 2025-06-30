#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=SIGLIP

DATASET=$1
SEED=$2

CFG=lr_1e-3_layers_4_epochs_5_base2new
SHOTS=16

PROJ_LAYERS=1
BATCH_SIZE=$3
LR=$4
EPOCHS=$5
ALPHA=$6
TEMP=$7
GPU=$8
BACKBONE="ViT-B/16"

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${EPOCHS}_${ALPHA}_${TEMP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr        ${LR} \
    --ep        ${EPOCHS} \
    --alpha     ${ALPHA} \
    --temp      ${TEMP} \
    --layers    ${PROJ_LAYERS} \
    --trainer   ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/QKMASK/${CFG}.yaml \
    --output-dir ${DIR} \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr       ${LR} \
    --ep        ${EPOCHS} \
    --alpha     ${ALPHA} \
    --temp      ${TEMP} \
    --layers    ${PROJ_LAYERS} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/QKMASK/${CFG}.yaml \
    --output-dir ${DIR} \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi