#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/ghazi.ahmad/DATA"
TRAINER=QKMASK

DATASET=$1
SEED=$2

CFG=lr_1e-3_layers_4_epochs_5_base2new
SHOTS=16

PROJ_LAYERS=1
BATCH_SIZE=$3
LR=$4
EPOCHS=$5
ALPHA=$6
LR1=$7

echo "alpha: $ALPHA"

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${EPOCHS}_${ALPHA}_${LR1}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr        ${LR} \
    --ep        ${EPOCHS} \
    --alpha     ${ALPHA} \
    --layers    ${PROJ_LAYERS} \
    --lr1       ${LR1} \
    --trainer   ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr       ${LR} \
    --ep        ${EPOCHS} \
    --alpha     ${ALPHA} \
    --layers    ${PROJ_LAYERS} \
    --lr1       ${LR1} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi