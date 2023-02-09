#!/usr/bin/bash

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif
OVERLAY_FILE=/scratch/km3888/singularity_forge/3d.ext3:ro

singularity exec --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash -c 'source /ext3/env.sh; tensorboard --logdir ~/general_clip_forge/runs --port=6166'
