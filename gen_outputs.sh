#!/bin/bash


# Set the environment variable WEIGHTS_DIR to the first argument
WEIGHTS_DIR=$1

FULL_WEIGHTS_DIR="km3888@greene.hpc.nyu.edu:/scratch/km3888/queries/$WEIGHTS_DIR"

# scp -r $FULL_WEIGHTS_DIR /scratch/km3888/clip_forge_weights/

#Iterate through all the files in the directory /scratch/km3888/clip_forge_weights/$WEIGHTS_DIR
#For each file, run the gen_outputs.py script with the checkpoint_dir set to the directory of the file
#and the output_dir set to the visuals/$WEIGHTS_DIR directory
mkdir visuals/$WEIGHTS_DIR/
for file in /scratch/km3888/clip_forge_weights/$WEIGHTS_DIR/*
do
    echo "Processing $file"
    # Take the filename without the path
    file_name=$(basename $file)
    ~/miniconda3/envs/clip_forge/bin/python gen_outputs.py --checkpoint_dir /scratch/km3888/clip_forge_weights/$WEIGHTS_DIR/$file_name/ --output_dir visuals/$WEIGHTS_DIR/$file_name/
done