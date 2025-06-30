#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=RAda

DATASET=$1
SEED=$2
LR=$4
BATCH_SIZE=$3
PROJ_LAYERS=1

CFG=lr_1e-3_layers_4_epochs_5_base2new
SHOTS=16
LOADEP=$5
ALPHA=$6
TEMP=$7
GPU=$8
SUB=new
# BACKBONE="ViT-B/16"
# BACKBONE="RN50"
BACKBONE="ViT-L/14"



COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${BATCH_SIZE}_${LR}_${LOADEP}_${ALPHA}_${TEMP}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr       ${LR} \
    --alpha     ${ALPHA} \
    --temp      ${TEMP} \
    --layers    ${PROJ_LAYERS} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs       ${BATCH_SIZE} \
    --lr       ${LR} \
    --alpha     ${ALPHA} \
    --temp      ${TEMP} \
    --layers    ${PROJ_LAYERS} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi