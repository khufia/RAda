#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=QKMASK

DATASET=$1
SEED=$2
BATCH_SIZE=$3
LR=$4
SHOTS=16
CFG=lr_1e-4_layers_4_epochs_5_cross
SHOTS=16
LOADEP=$5
ALPHA=$6
TEMP=$7
gpu=$8



DIR=output/evaluation/${DATASET}/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${LOADEP}_${ALPHA}_${TEMP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs ${BATCH_SIZE} \
    --lr ${LR} \
    --ep ${LOADEP} \
    --alpha ${ALPHA} \
    --temp ${TEMP}  \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${LOADEP}_${ALPHA}_${TEMP}/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only
fi