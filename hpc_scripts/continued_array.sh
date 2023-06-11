#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=continue_training
#SBATCH --output=continue_training.out
#SBATCH --mail-user=km3888@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/km3888/general_clip_forge/slurm/job%A_%a.out
#SBATCH --error=/home/km3888/general_clip_forge/slurm/job%A_%a.err
#SBATCH --gres=gpu:a100:3
#SBATCH --array=0-1

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
python hpc_scripts/job_array_bin.py --setting ${SLURM_ARRAY_TASK_ID} \
--slurm_id ${SLURM_ARRAY_JOB_ID} \
--num_voxels 128 --gpu 0 --query_array "airplane" \
--init og_init --init_base /scratch/km3888/inits \
--query_dir /scratch/km3888/queries --improved_contrast \
"