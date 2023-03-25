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

from continued_utils import query_arrays, make_writer, get_networks, get_local_parser, get_clip_model,get_text_embeddings,get_type,make_init_dict, get_prompts, generate_gpt_prompts

class Wrapper(nn.Module):
    def __init__(self, args, clip_model, autoencoder, latent_flow_model, renderer, resizer, query_array):
        super(Wrapper, self).__init__()
        self.clip_model = clip_model

        #freeze clip model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.autoencoder = autoencoder
        self.latent_flow_model = latent_flow_model

        self.renderer = renderer

        #freeze renderer
        for param in self.renderer.parameters():
            param.requires_grad = False
        self.renderer.eval()

        self.resizer = resizer
        self.query_array = query_array
        self.args = args

    def clip_loss(self, text_features, iter):
        # out_3d = self.gen_shapes(text_features)
        out_3d = gen_shapes(self.query_array, self.args, self.autoencoder, self.latent_flow_model, text_features)
        out_3d_soft = torch.sigmoid(self.args.beta*(out_3d-self.args.threshold))#.clone()

        ims = self.renderer(out_3d_soft,orthogonal=self.args.orthogonal).double()
        ims = self.resizer(ims)
        im_samples = ims.view(-1,3,224,224)
        
        im_embs = self.clip_model.encode_image(ims)
        if self.args.renderer=='ea':
            #baseline renderer gives 3 dimensions
            text_features=text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512)
        
        losses=-1*torch.cosine_similarity(text_features,im_embs)

       
        return losses, im_samples

    def forward(self, text_features, iter):
        return self.clip_loss(text_features, iter)
       
def evaluate_true_voxel(out_3d_hard,args,clip_model,text_features,i,query_array):
    
    # code for saving the "true" voxel image
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    n_unique = len(set(query_array))
    num_shapes = min([n_unique, 3])
    for shape in range(num_shapes):
        save_path = '/scratch/mp5847/queries/%s/sample_%s_%s.png' % (args.id,i,shape)
        voxel_save(out_3d_hard[shape].squeeze().detach().cpu(), None, out_file=save_path)
        # load the image that was saved and transform it to a tensor
        voxel_im = PIL.Image.open(save_path).convert('RGB')
        voxel_tensor = T.ToTensor()(voxel_im)
        voxel_ims.append(voxel_tensor.unsqueeze(0))
    
    voxel_ims = torch.cat(voxel_ims,0)
    grid = torchvision.utils.make_grid(voxel_ims, nrow=num_shapes)

    for shape in range(num_shapes):
        save_path = '/scratch/mp5847/queries/%s/sample_%s_%s.png' % (args.id,i,shape)
        # os.remove(save_path)

    if args.use_tensorboard:
        args.writer.add_image('voxel image', grid, i)
    # #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_ims)

    # get CLIP embedding
    # voxel_image_embedding = visual_model(voxel_tensor.to(args.device).type(visual_model_type))
    voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
    voxel_similarity = torch.cosine_similarity(text_features[:num_shapes], voxel_image_embedding).mean()
    return voxel_similarity

def gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features):
    autoencoder.train()#TODO don't do this in eval
    latent_flow_model.eval() # has to be in .eval() mode for the sampling to work (which is bad but whatever)
    
    voxel_size = args.num_voxels
    batch_size = len(query_array)
    
    #hard code for now
    batch_size = len(text_features)
        
    shape = (voxel_size, voxel_size, voxel_size)
    p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(text_features.device)
    query_points = p.expand(batch_size, *p.size())
        
    noise = torch.Tensor(batch_size, args.emb_dims).normal_().to(text_features.device)
    decoder_embs = latent_flow_model.sample(text_features.device, batch_size, noise=noise, cond_inputs=text_features)

    out_3d = autoencoder(decoder_embs, query_points).view(batch_size, voxel_size, voxel_size, voxel_size).to(text_features.device)
    return out_3d

def do_eval(renderer,query_array,args,clip_model,autoencoder,latent_flow_model,resizer,iter,text_features):
    with torch.no_grad():
      out_3d = gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features)
    #save out_3d to numpy file
    # with open(f'out_3d/{args.learning_rate}_{args.query_array}/out_3d_{iter}.npy', 'wb') as f:
    #     np.save(f, out_3d.cpu().detach().numpy())
    out_3d_hard = out_3d.detach() > args.threshold
    # rgbs_hard = renderer.render(out_3d_hard.float(),orthogonal=args.orthogonal).double().to(args.device)
    rgbs_hard = renderer(out_3d_hard.float(),orthogonal=args.orthogonal).to(args.device)
    rgbs_hard = resizer(rgbs_hard)
    hard_im_embeddings = clip_model.encode_image(rgbs_hard)
    if args.renderer=='ea':
        #baseline renderer gives 3 dimensions
        hard_loss = -1*torch.cosine_similarity(text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512),hard_im_embeddings).mean()
    else:
        hard_loss = -1*torch.cosine_similarity(text_features,hard_im_embeddings).mean()
        
    #write to tensorboard
    voxel_render_loss = -1* evaluate_true_voxel(out_3d_hard,args,clip_model,text_features,iter,query_array)
    if args.use_tensorboard:
        args.writer.add_scalar('Loss/hard_loss', hard_loss, iter)
        args.writer.add_scalar('Loss/voxel_render_loss', voxel_render_loss, iter)

def test_train(args,clip_model,autoencoder,latent_flow_model,renderer):    
    resizer = T.Resize(224)

    losses = []
    if args.query_array in query_arrays:
        query_array = query_arrays[args.query_array]
    else:
        query_array = [args.query_array]

    #check if file json_name.json exists
    if not os.path.exists("json_name.json"):
        print("GPT-3 prompt file not found. Generating prompts...")
        generate_gpt_prompts(["wineglass","spoon","fork","knife","screwdriver","hammer","pencil","screw","plate","mushroom","umbrella","thimble","sombrero","sandal"])
    else:
        print("GPT-3 prompt file found. Skipping prompt generation...")
    
    query_array = query_array*args.num_views

    # make directory for saving images with name of the text query using os.makedirs
    if not os.path.exists('/scratch/mp5847/queries/%s' % args.id):
        os.makedirs('/scratch/mp5847/queries/%s' % args.id)   

    wrapper = Wrapper(args, clip_model, autoencoder, latent_flow_model, renderer, resizer, query_array)

    wrapper = nn.DataParallel(wrapper).to(args.device)

    wrapper_optimizer = optim.Adam(wrapper.parameters(), lr=args.learning_rate)

    for iter in range(20000):
        text_features = get_text_embeddings(args,clip_model, query_array).detach()

        if args.switch_point is not None and iter == args.switch_point:
            args.renderer = 'nvr+'
            renderer = NVR_Renderer(args, args.device)

        if not iter%100:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    do_eval(wrapper.module.renderer, query_array, args, wrapper.module.clip_model, wrapper.module.autoencoder, wrapper.module.latent_flow_model, wrapper.module.resizer, iter, text_features)
                    
        if not (iter%5000) and iter!=0:
            # save encoder and latent flow network
            torch.save(wrapper.module.latent_flow_model.state_dict(), '/scratch/mp5847/queries/%s/flow_model_%s.pt' % (args.id,iter))
            torch.save(wrapper.module.autoencoder.state_dict(), '/scratch/mp5847/queries/%s/aencoder_%s.pt' % (args.id,iter))

        wrapper_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss, im_samples = wrapper(text_features, iter)
            loss = loss.mean()
                    
        if(iter % 100 == 0):
            print('iter: ', iter, 'loss: ', loss.item())

        loss.backward()
        losses.append(loss.detach().item())
        
        if args.use_tensorboard and not iter%50:
            if not iter%50:
                grid = torchvision.utils.make_grid(im_samples, nrow=3)
                args.writer.add_image('images', grid, iter)
            args.writer.add_scalar('Loss/train', loss.item(), iter)            
        
        wrapper_optimizer.step()
        if not iter:
            print('finished first iter')
    
    #save latent flow and AE networks
    torch.save(wrapper.module.state_dict(), '/scratch/mp5847/queries/%s/final_flow_model.pt' % args.id)
    torch.save(wrapper.module.autoencoder.encoder.state_dict(), '/scratch/mp5847/queries/%s/final_aencoder.pt' % args.id)
    
    print(losses)


def main(args):
    args.writer = make_writer(args)
    args.id = args.writer.log_dir.split('runs/')[-1]
    
    device, gpu_array = helper.get_device(args)
    args.device = device
    
    print("Using device: ", device)
    args, clip_model = get_clip_model(args) 

    init_dict = make_init_dict()[args.init]
    net,latent_flow_network = get_networks(args,init_dict)

    param_dict={'device':args.device,'cube_len':args.num_voxels}
    if args.renderer == 'ea' or args.switch_point is not None:
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