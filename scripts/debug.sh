cd /home/mp5847/src/general_clip_forge; export PYTHONPATH="$PYTHONPATH:$PWD"; 

python continued_training.py --num_voxels 128 --learning_rate 01e-05 \
    --gpu 0 --beta 150.0 --query_array "fork" --num_views 1 \
    --checkpoint_dir_base /scratch/mp5847/general_clip_forge/exps/models/autoencoder \
    --checkpoint best_iou --checkpoint_dir_prior \
    /scratch/mp5847/general_clip_forge/exps/models/prior --checkpoint_nf best --renderer nvr+