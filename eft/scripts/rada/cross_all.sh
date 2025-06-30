#!/bin/bash

datasets=('imagenet_a' "imagenet_sketch" "imagenet_r" "imagenetv2" "sun397" "stanford_cars" "caltech101" "dtd" "eurosat"
            "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "ucf101")

SEED=1
BS=$1
LR=$2
EP=$3
ALPHA=$4
TEMP=$5
gpu=$6

echo "Evaluating on Novel Classes"

for dataset in "${datasets[@]}"
do
    echo "Running $dataset"
    bash scripts/qkmask/xd_test_maple.sh $dataset ${SEED} ${BS} ${LR} ${EP} ${ALPHA} ${TEMP} ${gpu}
    
    # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo "Script for $dataset completed successfully."
    else
        echo "Script for $dataset failed. Exiting."
        exit 1  # Exit the loop if any script fails
    fi
done