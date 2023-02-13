#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=continue_training
#SBATCH --output=continue_training.out
#SBATCH --mail-user=km3888@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/km3888/general_clip_forge/slurm/job%A_%a.out
#SBATCH --error=/home/km3888/general_clip_forge/slurm/job%A_%a.err
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5

module purge

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/km3888/singularity_forge/3d.ext3:ro

sleep $(( (RANDOM%10) + 1 ))
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo ${SLURM_ARRAY_TASK_ID}

singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash \
-c "source /ext3/miniconda3/bin/activate;\
python job_array.py --setting ${SLURM_ARRAY_TASK_ID}\
 --num_voxels 128 --gpu 0 --query_array "airplane" \
--checkpoint_dir_base ./exps/models/autoencoder \
--checkpoint best_iou --checkpoint_dir_prior \
./exps/models/prior/ --checkpoint_nf best \
"
