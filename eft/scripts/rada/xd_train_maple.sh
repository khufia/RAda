#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=QKMASK

DATASET=$1
SEED=$2
BATCH_SIZE=$3
LR=$4
EPOCHS=$5
ALPHA=$6
TEMP=$7
gpu=$8

CFG=lr_1e-4_layers_4_epochs_5_cross
SHOTS=16


DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${EPOCHS}_${ALPHA}_${TEMP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs ${BATCH_SIZE} \
    --lr ${LR} \
    --ep ${EPOCHS} \
    --alpha ${ALPHA} \
    --temp ${TEMP} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi