import os

import torch
import torch.optim as optim

from networks.zeroforge_model import ZeroForge

import clip

from rendering.nvr_renderer import NVR_Renderer
from rendering.baseline_renderer import BaselineRenderer

import torchvision
import torchvision.transforms as T

import PIL
import numpy as np
import torch.nn as nn
from zf_utils import make_writer, get_networks,\
                    get_local_parser, get_clip_model,get_query_array,\
                    get_text_embeddings, set_seed, save_networks, plt_render, \
                    voxel_save,get_local_parser,get_device
from zf_training import clip_loss
import PIL
import pdb


def main(args):
    set_seed(args.seed)

    args.writer = make_writer(args)
    args.id = args.writer.log_dir.split('runs/')[-1]
    
    device, _ = get_device(args)
    args.device = device
    
    print("Using device: ", device)
    args, clip_model = get_clip_model(args) 

    net,latent_flow_network = get_networks(args)

    # make directory for saving images with name of the text query using os.makedirs
    if not os.path.exists(f'{args.log_dir}/{args.id}'):
        os.makedirs(f'{args.log_dir}/{args.id}')

    # Baseline ea renderer uses ray-trace (instead of nn) to get object silhouette
    # Doesn't give as good of renderings as NVR+ but can be useful for debugging/testing
    if args.renderer == 'ea':
        param_dict={'device':args.device,'cube_len':args.num_voxels}
        renderer=BaselineRenderer('absorption_only',param_dict)
    elif args.renderer == 'nvr+':
        renderer = NVR_Renderer(args, args.device)

    train_zf(args,clip_model,net,latent_flow_network,renderer)
    
if __name__=="__main__":
    parser=get_local_parser(mode=None)
    parser.add_argument("--color",type=str,default="red")

    args = parser.parse_args()
    
    main(args)

