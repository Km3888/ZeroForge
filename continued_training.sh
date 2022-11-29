#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate clip_forge

python continued_training.py --num_voxels 64 --gpu 0 --beta 50.0 --query_array "fork" --num_views 4 --checkpoint_dir_base ./exps/models/autoencoder --checkpoint best_iou --checkpoint_dir_prior ./exps/models/prior/ --checkpoint_nf best