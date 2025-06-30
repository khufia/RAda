feature_dir=clip_feat

datasets=("DescribableTextures" "EuroSAT" 'ImageNet' "SUN397" "StanfordCars" "Caltech101"  
            "FGVCAircraft" "Food101" "OxfordFlowers" "OxfordPets" "UCF101")


for dataset in "${datasets[@]}"
do
    python linear_probe.py \
    --dataset ${dataset} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 1
done
