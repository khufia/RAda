#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=lr_1e-3_layers_4_epochs_5_base2new # rn50, rn101, vit_b32 or vit_b16
GPU=$2

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/QKMASK/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
DATASET.NUM_SHOTS 16 \
DATASET.SUBSAMPLE_CLASSES new