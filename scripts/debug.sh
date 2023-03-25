cd /home/mp5847/src/general_clip_forge; export PYTHONPATH="$PYTHONPATH:$PWD"; 

python continued_training.py --num_voxels 128 --learning_rate 01e-05 \
    --gpu 0 --beta 150.0 --query_array "four" --num_views 5 \
    --init og_init --renderer nvr+ --nvr_renderer_checkpoint "/scratch/mp5847/general_clip_forge/nvr_plus.pt" \
    --use_gpt_prompts