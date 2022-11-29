#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate clip_forge

python train_autoencoder.py --dataset_path /datasets/ShapeNet/ --emb_dims 256 --gpu 0

python train_post_clip.py  --dataset_path /datasets/ShapeNet/ --checkpoint best_iou  --num_views 1 --num_blocks 8 --num_hidden 2048 --emb_dims 256 --gpu 0 --text_query "a chair" "a limo" "a jet plane"

