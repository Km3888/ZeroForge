import os
import gc
import time

import torch
import torch.optim as optim

from networks.wrapper import Wrapper

from test_post_clip import voxel_save

from train_post_clip import get_local_parser, get_clip_model

from utils import helper

from rendering.nvr_renderer import NVR_Renderer
from rendering.baseline_renderer import BaselineRenderer

import torchvision
import torchvision.transforms as T

import PIL
import sys
import numpy as np
import torch.nn as nn

from continued_utils import query_arrays, make_writer, get_networks, get_local_parser, get_clip_model,get_text_embeddings,make_init_dict
import PIL

       
def evaluate_true_voxel(out_3d_hard,args,clip_model,text_features,i,query_array):
    # code for saving the "true" voxel image
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    n_unique = len(set(query_array))
    num_shapes = min([n_unique, 3])
    for shape in range(num_shapes):
        save_path = '%s/%s/sample_%s_%s.png' % (args.query_dir,args.id,i,shape)
        voxel_save(out_3d_hard[shape].squeeze().detach().cpu(), None, out_file=save_path)
        # load the image that was saved and transform it to a tensor
        voxel_im = PIL.Image.open(save_path).convert('RGB')
        voxel_tensor = T.ToTensor()(voxel_im)
        voxel_ims.append(voxel_tensor.unsqueeze(0))
    
    voxel_ims = torch.cat(voxel_ims,0)
    grid = torchvision.utils.make_grid(voxel_ims, nrow=num_shapes)

    for shape in range(num_shapes):
        save_path = '%s/%s/sample_%s_%s.png' % (args.query_dir,args.id,i,shape)
        # os.remove(save_path)

    if args.use_tensorboard:
        args.writer.add_image('voxel image', grid, i)
    # #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_ims)

    # get CLIP embedding
    # voxel_image_embedding = clip_model(voxel_tensor.to(args.device).type(clip_model_type))
    voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
    voxel_similarity = torch.cosine_similarity(text_features[:num_shapes], voxel_image_embedding).mean()
    return voxel_similarity


def save_images(rgbs_hard,iter,args,query_array):
    rgbs_hard = rgbs_hard.view(-1,3,224,224)
    # save each image separately using args.id and PIL
    # creat a folder for each query array if it doesn't exist
    if not os.path.exists(f'/scratch/km3888/queries/out_images/{args.id}'):
        os.makedirs(f'/scratch/km3888/queries/out_images/{args.id}')
    already_saved = set()
    for i in range(len(rgbs_hard)):
        if query_array[i] in already_saved:
            continue
        rgbs_hard[i] = (rgbs_hard[i] - rgbs_hard[i].min()) / (rgbs_hard[i].max() - rgbs_hard[i].min())
        rgbs_hard[i] = rgbs_hard[i].mul(255).clamp(0, 255).byte()
        rgbs_hard_i = rgbs_hard[i].permute(1,2,0)
        rgbs_hard_i = rgbs_hard_i.cpu().numpy()
        rgbs_hard_i = PIL.Image.fromarray(rgbs_hard_i.astype(np.uint8))
        rgbs_hard_i.save(f'/scratch/km3888/queries/out_images/{args.id}/{iter}_{query_array[i]}.png')
        already_saved.add(query_array[i])
    
def do_eval(query_array,args,iter,text_features,best_hard_loss,wrapper):    
    with torch.no_grad():
      out_3d_hard, rgbs_hard, hard_im_embeddings = wrapper(text_features,hard=True)
    #save out_3d to numpy file
    # with open(f'out_3d/{args.learning_rate}_{args.query_array}/out_3d_{iter}.npy', 'wb') as f:
    #     np.save(f, out_3d.cpu().detach().numpy())
    if args.renderer=='ea':
        #baseline renderer gives 3 dimensions
        hard_loss = -1*torch.cosine_similarity(text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512),hard_im_embeddings).mean()
    else:
        hard_loss = -1*torch.cosine_similarity(text_features,hard_im_embeddings).mean()
        
    #write to tensorboard
    # voxel_render_loss = -1* evaluate_true_voxel(out_3d_hard,args,clip_model,text_features,iter,query_array)
    if args.use_tensorboard:
        args.writer.add_scalar('Loss/hard_loss', hard_loss, iter)
        # args.writer.add_scalar('Loss/voxel_render_loss', voxel_render_loss, iter)

    if hard_loss<best_hard_loss:
        save_images(rgbs_hard,iter,args,query_array)
        best_hard_loss = hard_loss
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_hard_loss

def evaluate_true_voxel(out_3d_hard,args,clip_model,text_features,i,query_array):
    # code for saving the "true" voxel image
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    n_unique = len(set(query_array))
    num_shapes = min([n_unique, 3])
    for shape in range(num_shapes):
        save_path = '/scratch/km3888/queries/%s/sample_%s_%s.png' % (args.id,i,shape)
        voxel_save(out_3d_hard[shape].squeeze().detach().cpu(), None, out_file=save_path)
        # load the image that was saved and transform it to a tensor
        voxel_im = PIL.Image.open(save_path).convert('RGB')
        voxel_tensor = T.ToTensor()(voxel_im)
        voxel_ims.append(voxel_tensor.unsqueeze(0))
    
    voxel_ims = torch.cat(voxel_ims,0)
    grid = torchvision.utils.make_grid(voxel_ims, nrow=num_shapes)

    for shape in range(num_shapes):
        save_path = '/scratch/km3888/queries/%s/sample_%s_%s.png' % (args.id,i,shape)
        os.remove(save_path)

    if args.use_tensorboard:
        args.writer.add_image('voxel image', grid, i)
    # #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_ims)
    # get CLIP embedding
    voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
    voxel_similarity = torch.cosine_similarity(text_features[:num_shapes], voxel_image_embedding).mean()
    return voxel_similarity

def clip_loss(im_embs,text_features,args,query_array):
    loss = -1*torch.cosine_similarity(text_features,im_embs).mean()
    
    #normalize im_embs
    im_embs = im_embs / im_embs.norm(dim=-1, keepdim=True)
    
    #compute all pair cosine similarity between im_embs and text_features
    cos_sim = torch.mm(im_embs, text_features.t())
    if args.improved_contrast:
        k = len(set(query_array))
        n = cos_sim.shape[0]
        mask = torch.zeros_like(cos_sim)

        for i in range(1,args.num_views):
            upper_diag = torch.diag(torch.ones(n - k*i), diagonal=k*i).to(args.device)
            lower_diag = torch.diag(torch.ones(n - k*i), diagonal=-1*k*i).to(args.device)
            mask = mask +  upper_diag + lower_diag
        mask = 1 - mask
        mask = mask.to(args.device)
        cos_sim = cos_sim * mask
    probs = torch.softmax(args.temp*cos_sim, dim=1)
    log_probs = torch.log(probs)
    diag_terms = log_probs.diag()
    
    if args.log_contrast:
        diag_terms = torch.log(diag_terms)
    contrast_loss = -1*diag_terms.mean()
    #compute loss
    if args.all_contrast:
        return contrast_loss,contrast_loss
    train_loss = loss + contrast_loss * args.contrast_lambda
    if args.std_coeff>0:
        std_loss = calc_std(im_embs)
        train_loss = train_loss + args.std_coeff * std_loss
    return train_loss,loss

def calc_kl(out_3d,sphere):
    eps = 01e-04
    loss_kl = torch.log(sphere * (sphere/(out_3d+eps)) + eps).mean()
    return loss_kl

def calc_std(im_embs):
    std_ims = torch.sqrt(im_embs.var(dim=0)+0.0001)
    std_loss = -1*torch.mean(std_ims)
    return std_loss

def make_sphere(args):
    num_voxels = args.num_voxels
    grid_xyz = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, num_voxels),
        torch.linspace(-1, 1, num_voxels),
        torch.linspace(-1, 1, num_voxels),
    ), -1)
    with torch.no_grad():
        radius = torch.sqrt((grid_xyz.unsqueeze(0).unsqueeze(1) ** 2).sum(-1)).squeeze()
        radius_mask = radius <= 1.0
        spherical_prior = torch.zeros((num_voxels,num_voxels,num_voxels)).to(args.device)
        spherical_prior[radius_mask] = 1.0
        spherical_prior[~radius_mask] = 0.0
    return spherical_prior

def test_train(args,clip_model,autoencoder,latent_flow_model,renderer):    
    resizer = T.Resize(224)

    losses = []
    if args.query_array in query_arrays:
        query_array = query_arrays[args.query_array]
    else:
        query_array = [args.query_array]
    
    query_array = query_array*args.num_views

    print('query array:',query_array)
    text_features = get_text_embeddings(args,clip_model,query_array).detach()
    # make directory for saving images with name of the text query using os.makedirs
    if not os.path.exists(f'{args.query_dir}/{args.id}'):
        os.makedirs(f'{args.query_dir}/{args.id}')

    wrapper = Wrapper(args, clip_model, autoencoder, latent_flow_model, renderer, resizer, query_array)

    wrapper = nn.DataParallel(wrapper).to(args.device)

    start_time = time.time()
    best_hard_loss = float('inf')
    wrapper_optimizer = optim.Adam(wrapper.parameters(), lr=args.learning_rate)

    spherical_prior = make_sphere(args)
    
    for iter in range(20000):
        if args.use_gpt_prompts:
            text_features = get_text_embeddings(args,clip_model, query_array).detach()

        if not iter%500:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    wrapper.module.autoencoder.eval()
                    do_eval(query_array, args,iter, text_features,best_hard_loss,wrapper)
                    
        if not (iter%5000) and iter!=0:
            # save encoder and latent flow network
            torch.save(wrapper.module.latent_flow_model.state_dict(), '%s/%s/flow_model_%s.pt' % (args.query_dir,args.id,iter))
            torch.save(wrapper.module.autoencoder.state_dict(), '%s/%s/aencoder_%s.pt' % (args.query_dir,args.id,iter))

        wrapper_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            wrapper.module.autoencoder.train()
            out_3d, im_samples, im_embs = wrapper(text_features, background = args.background)
            loss,strict_loss = clip_loss(im_embs, text_features, args, query_array)
            kl_penalty = calc_kl(out_3d, spherical_prior)
            loss += kl_penalty * args.kl_lambda
            #strict loss doesn't include contrative term
        if(iter % 100 == 0):
            print('iter: ', iter, 'loss: ', loss.item())

        loss.backward()
        losses.append(loss.detach().item())
        
        if args.use_tensorboard and not iter%50:
            if not iter%50:
                grid = torchvision.utils.make_grid(im_samples, nrow=3)
                args.writer.add_image('images', grid, iter)
            args.writer.add_scalar('Loss/train', loss.item(), iter)
            args.writer.add_scalar('Loss/strict_train_loss',strict_loss.item(),iter)          
        
        wrapper_optimizer.step()
        if not iter:
            print('train time:', time.time()-start_time)
        if not iter%500:
            print(iter)
            
    #save latent flow and AE networks
    torch.save(wrapper.module.state_dict(), '%s/%s/final_flow_model.pt' % (args.query_dir,args.id))
    torch.save(wrapper.module.autoencoder.encoder.state_dict(), '%s/%s/final_aencoder.pt' % (args.query_dir,args.id))
    
    print(losses)


def main(args):
    helper.set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.writer = make_writer(args)
    args.id = args.writer.log_dir.split('runs/')[-1]
    
    device, _ = helper.get_device(args)
    args.device = device
    
    print("Using device: ", device)
    args, clip_model = get_clip_model(args) 

    init_dict = make_init_dict()[args.init]
    net,latent_flow_network = get_networks(args,init_dict)

    param_dict={'device':args.device,'cube_len':args.num_voxels}
    if args.renderer == 'ea':
        renderer=BaselineRenderer('absorption_only',param_dict)
    elif args.renderer == 'nvr+':
        renderer = NVR_Renderer(args, args.device)

    test_train(args,clip_model,net,latent_flow_network,renderer)
    
if __name__=="__main__":
    args=get_local_parser()
    print('renderer %s' % args.renderer)
    import sys; sys.stdout.flush()

    main(args)
    
    sys.stdout.write(args.writer.log_dir[:5])
    sys.stdout.flush()
