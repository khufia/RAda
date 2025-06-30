# sh feat_extractor.sh
DATA=/l/users/zhiqiang.shen/ghazi/DATA
TRAINER=QKMASK

OUTPUT='./clip_feat/'
SEED=1
SHOTS=16

DIR='./clip_feat/'

datasets=("dtd" 'imagenet' "sun397" "stanford_cars" "caltech101" "eurosat"
            "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "ucf101")

for dataset in "${datasets[@]}"
do
    for SPLIT in train val test

    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${dataset}.yaml \
        --config-file ../configs/trainers/QKMASK/lr_1e-3_layers_4_epochs_5_base2new.yaml \
        --output-dir ${OUTPUT} \
        --eval-only \
        --trainer   ${TRAINER} \
        --output-dir ${DIR} 
    done
done
