import os
import torch
import torchvision.transforms as T
import argparse
import imageio

from train_post_clip import get_dataloader, experiment_name2, get_condition_embeddings, get_local_parser, get_clip_model
from test_post_clip import voxel_save
import clip
from networks import autoencoder, latent_flows
from utils import visualization
from PIL import Image

import io

def get_clip_embedding(query,clip_model):
    text = clip.tokenize([query]).to(args.device)
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def linear_interpolation(start_clip_embedding,end_clip_embedding,args):
    clip_embeddings = []
    for i in range(args.num_frames):
        interpolated_point = start_clip_embedding + (end_clip_embedding-start_clip_embedding)*i/args.num_frames
        if args.normalize_interpolated:
            interpolated_point/=interpolated_point.norm(dim=-1, keepdim=True)
        clip_embeddings.append(interpolated_point)
    return clip_embeddings


def save_voxel_images(net, latent_flow_model, text_features, args, save_path, resolution=64, num_figs_per_query=1):
    net.eval()
    latent_flow_model.eval()
    count = 1
    num_figs = num_figs_per_query
    with torch.no_grad():
        voxel_size = resolution
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
                
        ###########
        torch.manual_seed(5)
        mean_shape = torch.zeros(1, args.emb_dims).to(args.device) 
        noise = torch.Tensor(num_figs-1, args.emb_dims).normal_().to(args.device) 
        noise = torch.clip(noise, min=-1, max=1)
        noise = torch.cat([mean_shape, noise], dim=0)
        decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))

        out = net.decoding(decoder_embs, query_points)
        voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
        
        voxel_num = 0
        for voxel_in in voxels_out:
            voxel_save(voxel_in, None, out_file=save_path)
            voxel_num = voxel_num + 1


def generate_on_clip_embedding(args,text_features,ae_model,latent_flow_model,i,im_dir):
    ae_model.eval()
    latent_flow_model.eval()
    count = 1
    num_figs = 1
    save_loc='./'
    with torch.no_grad():
        voxel_size = 64
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
               
        noise = torch.Tensor(num_figs, args.emb_dims).normal_().to(args.device)
        decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))

        out = ae_model.decoding(decoder_embs, query_points)
        
        loc='%s/frame_%s' % (im_dir,i)
        if args.output_type == "Implicit":
            voxels_out = (out.view(voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            voxel_save(voxels_out,None,out_file=loc)
            # visualization.multiple_plot_voxel(voxels_out, save_loc=loc,show=False)
        elif args.output_type == "Pointcloud":
            pred = out.detach().cpu().numpy()
            visualization.multiple_plot(pred,  save_loc=save_loc +"{}_text_query.png".format(i))
    
def make_animation(im_dir,num_frames):
    images = []
    for i in range(num_frames+1):
        loc='%s/frame_%s.png' % (im_dir,i)
        images.append(imageio.imread(loc))
    imageio.mimsave('%s/animation.gif' % im_dir, images)

def main(args):
    # load models
    args,clip_model = get_clip_model(args)
    ae_model=autoencoder.get_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
    ae_model.load_state_dict(checkpoint['model'])
    ae_model.eval()
    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, args.device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
    checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
    latent_flow_network.load_state_dict(checkpoint['model'])
    latent_flow_network.eval()
    
    
    start_clip_embedding = get_clip_embedding(args.start,clip_model)
    end_clip_embedding = get_clip_embedding(args.finish,clip_model)
    
    #create output directory for images
    if not os.path.exists(args.im_save_dir):
        os.makedirs(args.im_save_dir)
    
    output_frames=[]
    for i,embedding in enumerate(linear_interpolation(start_clip_embedding,end_clip_embedding,args)):
        save_path = '%s/frame_%s' % (args.im_save_dir,i)
        save_voxel_images(ae_model,latent_flow_network,embedding,args,save_path)
        # output_frames.append(generate_on_clip_embedding(args,embedding,ae_model,latent_flow_network,i,args.im_save_dir))
    save_path = '%s/frame_%s' % (args.im_save_dir,args.num_frames)
    save_voxel_images(ae_model,latent_flow_network,end_clip_embedding,args,save_path)
    
    # make animation from output_frames
    make_animation(args.im_save_dir,args.num_frames)
        
if __name__=='__main__':
    # get args
    parser = get_local_parser(mode="parser")
    parser.add_argument('--start',type=str)
    parser.add_argument('--finish',type=str)
    parser.add_argument('--num_frames',type=int,default=10)
    parser.add_argument("--checkpoint_dir_base",  type=str, default="./exps/models/autoencoder", help='Checkpoint directory for autoencoder')
    parser.add_argument("--checkpoint_nf",  type=str, default="best", metavar='N', help='what is the checkpoint for nf')
    parser.add_argument("--checkpoint_dir_prior",  type=str, default="./exps/models/prior/", help='Checkpoint for prior')
    parser.add_argument("--normalize_interpolated", action="store_true", help='normalize interpolated points')
    parser.add_argument("--im_save_dir", type=str, default=None, help='where to save images used for animation')
    args=parser.parse_args()
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint is None:
        args.checkpoint="best_iou"
    if args.im_save_dir is None:
        args.im_save_dir="exps/animation/%s_%s_%s" % (args.start,args.finish,args.num_frames)
    if args.normalize_interpolated:
        args.im_save_dir+="_normalized"
    main(args)