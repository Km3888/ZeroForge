#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=2                # uses 1 compute core per task
#SBATCH --time=30:00
#SBATCH --mem=50GB
#SBATCH --job-name=continue_training
#SBATCH --output=continue_training.out
#SBATCH --mail-user=km3888@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/km3888/general_clip_forge/slurm/job%j.out
#SBATCH	--error=/home/km3888/general_clip_forge/slurm/job%j.err
#SBATCH --gres=gpu:a100:2

module purge

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/km3888/singularity_forge/3d.ext3:ro

singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash \
-c "source /ext3/miniconda3/bin/activate;\
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512';\
python continued_training.py --num_voxels 128 --learning_rate 01e-05 \
--gpu 0 --beta 150.0 --query_array "airplane" --num_views 8 \
--checkpoint_dir_base ./exps/models/autoencoder \
--checkpoint best_iou --checkpoint_dir_prior \
./exps/models/prior/ --checkpoint_nf best --renderer nvr+ \
"