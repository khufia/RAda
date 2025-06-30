#!/bin/bash

datasets=("sun397" "stanford_cars" "caltech101" "dtd" "eurosat"
            "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "ucf101")

# datasets=("eurosat")

SEED=$1
SUB=new
SHOTS=16
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx



for dataset in "${datasets[@]}"
do
    DIR=output/base2new/train_base/${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}


    LOG_FILE="${DIR}/log.txt"

    echo "Running $dataset"
    bash scripts/maple/base2new_train_maple.sh $dataset ${SEED} 
    
    # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo "Script for $dataset completed successfully."
                # Ensure log file exists before attempting extraction
        if [[ -f "$LOG_FILE" ]]; then
            # Extract accuracy from log file (removing '%' sign)
            acc=$(grep -oP '\* accuracy:\s+\K[0-9]+\.[0-9]+' "$LOG_FILE" | tail -1)

            if [[ -n "$acc" ]]; then
                base_accuracies+=("$acc")  # Store accuracy
            else
                echo "Warning: Could not extract accuracy from $LOG_FILE"
            fi
        else
            echo "Error: Log file not found at $LOG_FILE"
        fi
    else
        echo "Script for $dataset failed. Exiting."
        exit 1  # Exit the loop if any script fails
    fi
done

echo "Training Complete"

# -------------------
# ----------------------------------------
# ------------------------------------------------------------------



echo "Evaluating on Novel Classes"

for dataset in "${datasets[@]}"
do
    DIR=output/base2new/test_${SUB}/${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    LOG_FILE="${DIR}/log.txt"
    echo "Running $dataset"
    bash scripts/maple/base2new_test_maple.sh $dataset ${SEED} 
    
     # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo "Script for $dataset completed successfully."
                # Ensure log file exists before attempting extraction
        if [[ -f "$LOG_FILE" ]]; then
            # Extract accuracy from log file (removing '%' sign)
            acc=$(grep -oP '\* accuracy:\s+\K[0-9]+\.[0-9]+' "$LOG_FILE" | tail -1)

            if [[ -n "$acc" ]]; then
                novel_accuracies+=("$acc")  # Store accuracy
            else
                echo "Warning: Could not extract accuracy from $LOG_FILE"
            fi
        else
            echo "Error: Log file not found at $LOG_FILE"
        fi
    else
        echo "Script for $dataset failed. Exiting."
        exit 1  # Exit the loop if any script fails
    fi
done


# Compute and print mean accuracy
if [ ${#base_accuracies[@]} -gt 0 ]; then
    sum=0
    for acc in "${base_accuracies[@]}"; do
        sum=$(echo "$sum + $acc" | bc)  # Sum up accuracies
    done
    mean_acc=$(echo "scale=2; $sum / ${#base_accuracies[@]}" | bc)  # Compute mean
    echo "Base Accuracy: $mean_acc%"
else
    echo "No valid accuracy values found."
fi


# Compute and print mean accuracy
if [ ${#novel_accuracies[@]} -gt 0 ]; then
    sum=0
    for acc in "${novel_accuracies[@]}"; do
        sum=$(echo "$sum + $acc" | bc)  # Sum up accuracies
    done
    mean_acc=$(echo "scale=2; $sum / ${#novel_accuracies[@]}" | bc)  # Compute mean
    echo "Novel Accuracy: $mean_acc%"
else
    echo "No valid accuracy values found."
fi