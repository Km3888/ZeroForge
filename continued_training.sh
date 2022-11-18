#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate clip_forge

python continued_training.py --beta 25.0 --text_query "a knife" --checkpoint_dir_base ./exps/models/autoencoder --checkpoint best_iou --checkpoint_dir_prior ./exps/models/prior/ --checkpoint_nf best