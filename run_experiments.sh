#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate clip_forge

python train_autoencoder.py --dataset_path /datasets/ShapeNet/ 

python train_post_clip.py  --dataset_path /datasets/ShapeNet/ --checkpoint best_iou  --num_views 1 --text_query "a chair" "a limo" "a jet plane"

