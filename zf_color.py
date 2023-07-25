import os

import torch
import torch.optim as optim

from networks.zeroforge_model import ZeroForge
from networks.color_net import ConstantColor

import clip

from rendering.nvr_renderer import NVR_Renderer
from rendering.baseline_renderer import BaselineRenderer

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

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


def color_training(args,clip_model,net,latent_flow_model,renderer,color_net):
    
    resizer = T.Resize(224)

    query_array = get_query_array(args)
    text_features = get_text_embeddings(args,clip_model,query_array).detach()

    zf_model = ZeroForge(args, clip_model, net, latent_flow_model,color_net, renderer, resizer, query_array)
    zf_model = nn.DataParallel(zf_model).to(args.device)
    
    zf_optimizer = optim.Adam(zf_model.color_net.parameters(), lr=args.learning_rate)    
    zf_model.autoencoder.train()
    for i in range(1000):
        with torch.cuda.amp.autocast():
            zf_optimizer.zero_grad()
            out_3d, im_samples, im_embs,out_3d_gray = zf_model(text_features)
            out_3d.retain_grad()
            out_3d_gray.retain_grad()
            loss,similarity_loss = clip_loss(im_embs, text_features, args)
            os.makedirs(f'{args.log_dir}/{args.id}',exist_ok=True)

            # save im_samples
            im_samples_pil = im_samples.detach().cpu().squeeze()
            similarity_loss.backward()
            save_image(im_samples_pil, os.path.join(os.getcwd(),'im_samples_%s.png' % i))
            zf_optimizer.step()

def main(args):
    set_seed(args.seed)

    args.writer = make_writer(args,color=True)
    print(args.writer.log_dir)
    args.id = args.writer.log_dir.split('runs/')[-1] + '_2'
    
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
    
    color_net = ConstantColor().to(args.device)
    color_training(args,clip_model,net,latent_flow_network,renderer,color_net)
    
if __name__=="__main__":
    parser=get_local_parser(mode=None)
    parser.add_argument("--color",type=str,default="red")

    args = parser.parse_args()
    main(args)