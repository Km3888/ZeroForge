import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import os.path as osp
import logging
import gc

import torch
import torch.optim as optim

from networks import autoencoder, latent_flows
import clip
from test_post_clip import voxel_save

from train_post_clip import get_dataloader, experiment_name2, get_condition_embeddings, get_local_parser, get_clip_model
from train_autoencoder import parsing

from utils import helper
from utils import visualization
from utils import experimenter

from rendering.nvr_renderer import NVR_Renderer
from rendering.baseline_renderer import BaselineRenderer

import torchvision
from torchvision.utils import save_image
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

import PIL
import sys
import numpy as np
import torch.nn as nn

from continued_utils import query_arrays, make_writer, get_networks, get_local_parser, get_clip_model,get_text_embeddings,get_type

def clip_loss(args,query_array,visual_model,autoencoder,latent_flow_model,renderer,resizer,iter,text_features):
    out_3d = gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features)
    out_3d_soft = torch.sigmoid(args.beta*(out_3d-args.threshold))#.clone()
    
    ims = renderer(out_3d_soft,orthogonal=args.orthogonal).double()
    ims = resizer(ims)

    im_embs=visual_model(ims.type(visual_model_type))
    if args.renderer=='ea':
        #baseline renderer gives 3 dimensions
        text_features=text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512)

    losses=-1*torch.cosine_similarity(text_features,im_embs)
    loss = losses.mean()

    if args.use_tensorboard and not iter%50:
        im_samples= ims.view(-1,3,224,224)
        grid = torchvision.utils.make_grid(im_samples, nrow=3)
        args.writer.add_image('images', grid, iter)

    return loss

#TODO remove query_array param
def gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features):
    autoencoder.train()#TODO don't do this in eval
    latent_flow_model.eval() # has to be in .eval() mode for the sampling to work (which is bad but whatever)
    
    voxel_size = args.num_voxels
    batch_size = len(query_array)
        
    shape = (voxel_size, voxel_size, voxel_size)
    p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
    query_points = p.expand(batch_size, *p.size())
        
    noise = torch.Tensor(batch_size, args.emb_dims).normal_().to(args.device)
    decoder_embs = latent_flow_model.sample(batch_size, noise=noise, cond_inputs=text_features)

    out_3d = autoencoder(decoder_embs, query_points).view(batch_size, voxel_size, voxel_size, voxel_size).to(args.device)
    return out_3d

def do_eval(renderer,query_array,args,visual_model,autoencoder,latent_flow_model,resizer,iter,text_features):
    # import pdb; pdb.set_trace()
    # with torch.no_grad():
    out_3d = gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features)
    #save out_3d to numpy file
    # with open(f'out_3d/{args.learning_rate}_{args.query_array}/out_3d_{iter}.npy', 'wb') as f:
    #     np.save(f, out_3d.cpu().detach().numpy())
    out_3d_hard = out_3d.detach() > args.threshold
    # rgbs_hard = renderer.render(out_3d_hard.float(),orthogonal=args.orthogonal).double().to(args.device)
    # import pdb; pdb.set_trace()
    rgbs_hard = renderer(out_3d_hard.float(),orthogonal=args.orthogonal).to(args.device)
    rgbs_hard = resizer(rgbs_hard)
    # hard_im_embeddings = clip_model.encode_image(rgbs_hard)
    hard_im_embeddings = visual_model(rgbs_hard.type(visual_model_type))
    if args.renderer=='ea':
        #baseline renderer gives 3 dimensions
        hard_loss = -1*torch.cosine_similarity(text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512),hard_im_embeddings).mean()
    else:
        hard_loss = -1*torch.cosine_similarity(text_features,hard_im_embeddings).mean()
    #write to tensorboard
    # voxel_render_loss = -1* evaluate_true_voxel(out_3d_hard,args,visual_model,text_features,iter,query_array)
    if args.use_tensorboard:
        args.writer.add_scalar('Loss/hard_loss', hard_loss, iter)
        # args.writer.add_scalar('Loss/voxel_render_loss', voxel_render_loss, iter)

    rgbs_hard.to("cpu")
    out_3d_hard.to("cpu")
    out_3d.to("cpu")
    hard_loss.to("cpu")
    del rgbs_hard
    del out_3d_hard
    del out_3d
    del hard_loss
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_true_voxel(out_3d_hard,args,visual_model,text_features,i,query_array):
    # code for saving the "true" voxel image
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    import pdb; pdb.set_trace()
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
    voxel_image_embedding = visual_model(voxel_tensor.to(args.device).type(visual_model_type))
    voxel_similarity = torch.cosine_similarity(text_features[:num_shapes], voxel_image_embedding).mean()
    return voxel_similarity

def test_train(args,clip_model,autoencoder,latent_flow_model,renderer):    
    resizer = T.Resize(224)
    flow_optimizer=optim.Adam(latent_flow_model.parameters(), lr=args.learning_rate)
    net_optimizer=optim.Adam(autoencoder.parameters(), lr=args.learning_rate)

    losses = []
    if args.query_array in query_arrays:
        query_array = query_arrays[args.query_array]
    else:
        query_array = [args.query_array]
    query_array = query_array*args.num_views
    text_features = get_text_embeddings(args,clip_model,query_array).detach()
    # make directory for saving images with name of the text query using os.makedirs
    if not os.path.exists('/scratch/km3888/queries/%s' % args.id):
        os.makedirs('/scratch/km3888/queries/%s' % args.id)

    #remove text components from clip and free up memory
    visual_model = clip_model.visual
    del clip_model

    #set gradient of clip model to false
    for param in visual_model.parameters():
        param.requires_grad = False
    visual_model.eval()
    torch.cuda.empty_cache()

    global visual_model_type
    visual_model_type = get_type(visual_model)
    visual_model = nn.DataParallel(visual_model)

    for iter in range(20000):
        if args.switch_point is not None and iter == args.switch_point:
            args.renderer = 'nvr+'
            args.num_view = 10
            renderer = NVR_Renderer(args.device)
            renderer.model.to(args.device)

        import pdb; pdb.set_trace()
        if not iter%1000:
            with torch.cuda.amp.autocast():
                do_eval(renderer,query_array,args,visual_model,autoencoder,latent_flow_model,resizer,iter,text_features)
        
        if not (iter%5000) and iter!=0:
            #save encoder and latent flow network
            torch.save(latent_flow_model.state_dict(), '/scratch/km3888/queries/%s/flow_model_%s.pt' % (args.id,iter))
            torch.save(autoencoder.module.encoder.state_dict(), '/scratch/km3888/queries/%s/aencoder_%s.pt' % (args.id,iter))
            
        flow_optimizer.zero_grad()
        net_optimizer.zero_grad()
        
        import pdb; pdb.set_trace()
        with torch.cuda.amp.autocast():
            loss = clip_loss(args,query_array,visual_model,autoencoder,latent_flow_model,renderer,resizer,iter,text_features)        
        loss.backward()
        losses.append(loss.item())
        
        if args.use_tensorboard:
            args.writer.add_scalar('Loss/train', loss.item(), iter)
        
        flow_optimizer.step()
        net_optimizer.step()
        if not iter:
            print('finished first iter')
    
    #save latent flow and AE networks
    torch.save(latent_flow_model.state_dict(), '/scratch/km3888/queries/%s/final_flow_model.pt' % args.id)
    torch.save(autoencoder.module.encoder.state_dict(), '/scratch/km3888/queries/%s/final_aencoder.pt' % args.id)
    
    print(losses)


def main(args):
    args.writer = make_writer(args)
    args.id = args.writer.log_dir.split('runs/')[-1]
    
    device, gpu_array = helper.get_device(args)
    args.device = device
    
    print("Using device: ", device)
    args, clip_model = get_clip_model(args) 
    
    net,latent_flow_network = get_networks(args)
    
    param_dict={'device':args.device,'cube_len':args.num_voxels}
    if args.renderer == 'ea' or args.switch_point is not None:
        renderer=BaselineRenderer('absorption_only',param_dict)
    elif args.renderer == 'nvr+':
        renderer = NVR_Renderer(device)
        renderer = nn.DataParallel(renderer)
        renderer.to(args.device)
        # renderer.preprocessor = nn.DataParallel(renderer.preprocessor)
    net = nn.DataParallel(net)

    test_train(args,clip_model,net,latent_flow_network,renderer)
    
if __name__=="__main__":
    args=get_local_parser()
    print('renderer %s' % args.renderer)
    import sys; sys.stdout.flush()
    main(args)
    
    sys.stdout.write(args.writer.log_dir[:5])
    sys.stdout.flush()