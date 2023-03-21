#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:00
#SBATCH --mem=64GB
#SBATCH --job-name=clip_forge_prompt0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=clip_forge_prompt0_%j.out

module purge

singularity exec --nv \
    --overlay /scratch/km3888/singularity_forge/3d.ext3:ro \
    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c 'source /ext3/env.sh; cd /home/mp5847/src/general_clip_forge; export PYTHONPATH="$PYTHONPATH:$PWD"; \
      python continued_training.py --num_voxels 128 --learning_rate 01e-05 \
        --gpu 0 --beta 150.0 --query_array "mushroom" --num_views 10 \
        --checkpoint_dir_base /scratch/mp5847/general_clip_forge/exps/models/autoencoder \
        --checkpoint best_iou --checkpoint_dir_prior \
        /scratch/mp5847/general_clip_forge/exps/models/prior --checkpoint_nf best --renderer nvr+'




