#!/bin/bash
srun --time=47:59:00 --mem=128GB --gres=gpu:2 --nodes=1 --cpus-per-task=20 --pty /bin/bash

export SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif
export OVERLAY_FILE=/scratch/km3888/singularity_forge/3d.ext3:ro
singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash