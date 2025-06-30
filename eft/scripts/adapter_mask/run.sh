#!/bin/bash

datasets=('imagenet' "sun397" "stanford_cars" "caltech101" "dtd" "eurosat"
            "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "ucf101")

SEED=1
BS=$1
LR=$2
EP=$3
GPU=$4

for dataset in "${datasets[@]}"
do
    echo "Running $dataset"
    bash scripts/clip_adapter/base2new_train_maple.sh $dataset ${SEED} ${BS} ${LR} ${EP} ${GPU}
    
    # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo "Script for $dataset completed successfully."
    else
        echo "Script for $dataset failed. Exiting."
        exit 1  # Exit the loop if any script fails
    fi
done


echo "Training Complete"

echo "Evaluating on Novel Classes"

for dataset in "${datasets[@]}"
do
    echo "Running $dataset"
    bash scripts/clip_adapter/base2new_test_maple.sh $dataset ${SEED} ${BS} ${LR} ${EP} ${ALPHA} ${LR1}
    
    # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo "Script for $dataset completed successfully."
    else
        echo "Script for $dataset failed. Exiting."
        exit 1  # Exit the loop if any script fails
    fi
done