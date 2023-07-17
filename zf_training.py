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
import PIL
import pdb

# computes loss function for training ZeroForge
def clip_loss(im_embs,text_features,args):

    #start with simple similarity loss
    loss = -1*torch.cosine_similarity(text_features,im_embs).mean()
    
    #normalize im_embs
    im_embs = im_embs / im_embs.norm(dim=-1, keepdim=True)
    
    #compute all pairs cosine similarity between im_embs and text_features
    cos_sim = torch.mm(im_embs, text_features.t())
    
    #If the batch contains multiple instances of the same text query 
    #we want to mask out the similarity between identical instances
    k = args.unique #number of unique text queries
    if k<im_embs.shape[0]:
        n = cos_sim.shape[0]
        mask = torch.zeros_like(cos_sim)

        for i in range(1,args.num_views):
            upper_diag = torch.diag(torch.ones(n - k*i), diagonal=k*i).to(args.device)
            lower_diag = torch.diag(torch.ones(n - k*i), diagonal=-1*k*i).to(args.device)
            mask = mask +  upper_diag + lower_diag
        mask = 1 - mask
        mask = mask.to(args.device)
        cos_sim = cos_sim * mask
    
    #Compute contrastive loss
    probs = torch.softmax(args.temp*cos_sim, dim=1)
    log_probs = torch.log(probs)
    diag_terms = log_probs.diag()
    contrast_loss = -1*diag_terms.mean()
    
    #compute full training loss
    train_loss = loss + contrast_loss * args.contrast_lambda
    return train_loss,loss

def do_eval(query_array,args,iteration,text_features,zf_model,clip_model):
    query_array, args,iter, text_features,zf_model,clip_model
    #Collects training metrics and saves images for tensorboard
    with torch.no_grad():
      out_3d_hard, rgbs_hard, _ = zf_model(text_features,hard=True)
    num_shapes = out_3d_hard.shape[0]    
    if args.use_tensorboard:
        # matplotlib can gives better-quality 3D rendering but is not differentiable and requires binary voxels
        # we compute these renderings to give a better sense of the model's performance
        plt_ims = plt_render(out_3d_hard,iter,args)
        grid = torchvision.utils.make_grid(plt_ims, nrow=num_shapes)
        plt_ims = T.Resize((224,224))(plt_ims)
        plt_embs = clip_model.encode_image(plt_ims.to(args.device))
        render_sim_loss = -1*torch.cosine_similarity(text_features[:num_shapes], plt_embs).mean()
        args.writer.add_image('voxel image', grid, iteration)
        args.writer.add_scalar('render_sim_loss', render_sim_loss, iteration)
    

def train_zf(args,clip_model,autoencoder,latent_flow_model,renderer):    
    resizer = T.Resize(224)

    query_array = get_query_array(args)
    text_features = get_text_embeddings(args,clip_model,query_array).detach()

    zf_model = ZeroForge(args, clip_model, autoencoder, latent_flow_model, renderer, resizer, query_array)
    zf_model = nn.DataParallel(zf_model).to(args.device)
    zf_optimizer = optim.Adam(zf_model.parameters(), lr=args.learning_rate)    
    for iter in range(20000):
        # Main training loop for ZeroForge
        zf_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            zf_model.module.autoencoder.train()
            out_3d, im_samples, im_embs = zf_model(text_features)
            loss,similarity_loss = clip_loss(im_embs, text_features, args)
        loss.backward()
        zf_optimizer.step()

        # Log training metrics/save checkpoints
        if not iter%10:
            args.writer.add_scalar('Loss/train', loss.item(), iter)
            args.writer.add_scalar('Loss/similarity_loss',similarity_loss.item(),iter)          

        if (not iter%500):
            grid = torchvision.utils.make_grid(im_samples, nrow=3)
            args.writer.add_image('images', grid, iter)
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    zf_model.module.autoencoder.eval()
                    do_eval(query_array, args,iter, text_features,zf_model,clip_model)
                    
            if not (iter%5000) and iter!=0:
                save_networks(args,iter,zf_model)

    save_networks(args,iter,zf_model)

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
    args=get_local_parser()
    main(args)