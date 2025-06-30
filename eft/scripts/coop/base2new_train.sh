#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/zhiqiang.shen/ghazi/DATA"
TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=rn50_ctxv1
# CFG=rn101
SHOTS=16
CTP="end"
NCTX=16 # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=True  # class-specific context (False or True)
BACKBONE="ViT-B/16"
LR=2e-3
EP=10
# BACKBONE="RN101"
GPU=$3


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --lr ${LR}  \
    --ep ${EP}  \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    DATASET.NUM_SHOTS ${SHOTS}  \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --lr ${LR}  \
    --ep ${EP}  \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    MODEL.BACKBONE.NAME ${BACKBONE} \
    DATASET.NUM_SHOTS ${SHOTS}  \
    DATASET.SUBSAMPLE_CLASSES base
fi
