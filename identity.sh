#!/usr/bin/bash

TRANSFORMS=('')
TRANSFORMS_ELEMENTS=${#TRANSFORMS[@]}

# OBJECTS=("couch" "airplane" "lamp" "telephone" "boat" "speaker system" "chair" "cabinet" "table" "video display" "car" "bench" "rifle" "bicycle" "bus" "motorcycle" "train" "umbrella" "bottle" "wine glass" "refrigerator" "helicopter")
OBJECTS=("pistol")
OBJECTS_ELEMENTS=${#OBJECTS[@]}

eval "$(conda shell.bash hook)"
conda activate clip_forge 

for ((i=0;i<$TRANSFORMS_ELEMENTS;i++)); do
    for ((j=0;j<$OBJECTS_ELEMENTS;j++)); do
    conda activate clip_forge
    echo ${TRANSFORMS[${i}]}${OBJECTS[${j}]}
    python test_post_clip.py --checkpoint_dir_base "./exps/models/autoencoder" --checkpoint best_iou --checkpoint_nf best --experiment_mode save_voxel_on_query --checkpoint_dir_prior "./exps/models/prior" --text_query "${TRANSFORMS[${i}]} ${OBJECTS[${j}]}" --threshold 0.1 --output_dir "./exps/generalization_ablation/identity/"
done
done