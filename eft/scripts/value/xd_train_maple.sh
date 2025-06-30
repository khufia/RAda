#!/bin/bash

#cd ../..

# custom config
DATA="/l/users/ghazi.ahmad/DATA"
TRAINER=QKMASK

DATASET=$1
SEED=$2
BATCH_SIZE=$3
LR=$4
EPOCHS=$5

CFG=lr_1e-6_layers_1_epochs_4_cross
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --bs ${BATCH_SIZE} \
    --lr ${LR} \
    --ep ${EPOCHS} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi