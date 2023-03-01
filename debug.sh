# python job_array.py --num_voxels 128 --gpu 0 --query_array "airplane" --setting 1 \
#     --checkpoint_dir_base /scratch/mp5847/general_clip_forge/exps/models/autoencoder \
#     --checkpoint best_iou --checkpoint_dir_prior /scratch/mp5847/general_clip_forge/exps/models/prior  \
#     --checkpoint_nf best


python continued_training.py --num_voxels 128 --learning_rate 01e-05 \
   --gpu 0 --beta 150.0 --query_array "fork" --num_views 10 \
   --checkpoint_dir_base /scratch/mp5847/general_clip_forge/exps/models/autoencoder \
   --checkpoint best_iou --checkpoint_dir_prior \
   /scratch/mp5847/general_clip_forge/exps/models/prior --checkpoint_nf best --renderer nvr+