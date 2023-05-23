import numpy as np
from simple_3dviz import Mesh,Scene
# from simple_3dviz.window import show
from simple_3dviz.utils import render
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.utils import save_frame
from simple_3dviz import Lines
import argparse

from continued_utils import query_arrays,make_init_dict
import clip
from networks import autoencoder, latent_flows
import torch
from train_post_clip import get_clip_model
from utils import visualization
import tqdm
import os
from test_post_clip import id_to_sub_category

def save_voxels(voxels,out_path,query_array,thresh):
    # Load your voxel data as a NumPy array
    voxels = voxels.astype(np.bool8)  # Replace with the path to your voxel data
    # create directory for out_path if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Create a scene with a mesh and a line
    print('image out path:',out_path)
    for i in range(voxels.shape[0]):
        voxel = voxels[i]
        # l = Lines.from_voxel_grid(voxel, colors=(0, 0, 0.), width=0.01)
        m = Mesh.from_voxel_grid(voxel, colors=(0.8, 0, 0))
        # camera_position = [192, 192, 192]
        scene = Scene(size=[2048,2048])
        scene.add(m)
        # scene.add(l)
        #CameraTrajectory(Circle([0, 0.15, 0], [0, 0.15, -1.5], [0, 1, 0]), speed=0.01),
        # render(m,size=(800, 600),n_frames=1,behaviours=[SaveFrames("out.png")])
        scene.render()
        save_frame(out_path+"/%s_%s.png"% (query_array[i],thresh), scene.frame)

def get_networks(checkpoint_dir,init,zero_conv,args,iter=15000):
    init_dict = make_init_dict()[init]
    args.emb_dims = init_dict["emb_dim"]
    args.num_blocks = init_dict["num_blocks"]
    args.num_hidden = init_dict["num_hidden"]
    args.emb_dims = init_dict["emb_dim"]
    args.encoder_type = "Voxel_Encoder_BN"
    args.decoder_type = "Occ_Simple_Decoder"
    args.input_type = "Voxel"
    args.output_type = "Implicit"
    args.cond_emb_dim, args.device = ( 512, 'cuda:0')
    args.flow_type = "realnvp_half"
    net = autoencoder.EncoderWrapper(args).to(args.device)    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, args.device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    if  False and zero_conv:
        net.encoder.decoder = autoencoder.ZeroConvDecoder(net.encoder.decoder)
        net = net.to(args.device)
    checkpoint_nf_path = checkpoint_dir + "flow_model_%s.pt" % str(iter)
    # checkpoint_nf_path = "/scratch/km3888/inits/models/prior/best.pt" 
    checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
    # checkpoint = {k[18:]:v for k,v in checkpoint.items() if k.startswith('latent_flow_model')}
    latent_flow_network.load_state_dict(checkpoint)
    checkpoint_path = checkpoint_dir + "aencoder_%s.pt" % str(iter)
    # checkpoint_path = "/scratch/km3888/inits/models/autoencoder/best_iou.pt"
    checkpoint = torch.load(checkpoint_path,map_location=args.device)
    # checkpoint = {k[8:]:v for k,v in checkpoint.items()}
    net.load_state_dict(checkpoint)
    
    net.eval()
    args.clip_model_type = 'B-32'
    args,clip_model  = get_clip_model(args)

    #calculate total parameters in autoencoder and latent flow
    return net, latent_flow_network, clip_model

def gen_voxels(query_array, net, latent_flow_model,clip_model, output_dir):
    net.eval()
    latent_flow_model.eval()
    clip_model.eval()
    count = 1
    num_figs = 1
    
    voxels = []
    raw_voxels = []
    with torch.no_grad():
        voxel_size = 100
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
                
        for text_in in query_array:
            ##########
            text = clip.tokenize([text_in]).to(args.device)
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ###########
            torch.manual_seed(5)
            mean_shape = torch.zeros(1, args.emb_dims).to(args.device) 
            noise = torch.Tensor(num_figs-1, args.emb_dims).normal_().to(args.device) 
            noise = torch.clip(noise, min=-1, max=1)
            noise = torch.cat([mean_shape, noise], dim=0)
            decoder_embs = latent_flow_model.sample("cuda:0",num_samples=num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))

            out = net.encoder.decoding(decoder_embs, query_points)
            raw_voxels.append(out.view(num_figs, voxel_size, voxel_size, voxel_size).detach().cpu().numpy().squeeze())
            voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            voxels.append(voxels_out.squeeze())
    voxels = np.stack(voxels,axis=0)
    return voxels,raw_voxels

    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint_dir", type=str, default="/scratch/km3888/clip_forge_weights/33031346/q=easy_5_with_original_lr=1e-05_beta=200_gpu=0_baseline=False_v=128_k=2_r=nvr+_s=1_init=og_init_c=0.01_improved_temp=50/")
    parser.add_argument("--output_dir",type=str,default="voxels/visuals/33031346/q=easy_5_with_original_lr=1e-05_beta=200_gpu=0_baseline=False_v=128_k=2_r=nvr+_s=1_init=og_init_c=0.01_improved_temp=50")
    parser.add_argument("--init", type=str, default="og_init")
    parser.add_argument("--threshold", type=float, default=0.1)
    
    args = parser.parse_args()
    # get query array name from checkpoint_dir name
    # for example, q=easy_5_with_original_lr=1e-05_beta=200 gives query array name easy_5_with_original
    # 
    query_array_name = args.checkpoint_dir.split("=")[1][:-3]
    if query_array_name == "easy_5_with_original":
        query_array_name = "easy_five_with_original"
    if query_array_name=="superset" or query_array_name=="subset":
        query_array = list(id_to_sub_category.keys())
    else:
        query_array = query_arrays[query_array_name]
    # query_array = ["round table","square table","jagged knife","thick knife","thin knife","thick spoon","thin spoon","round chair"]
    zero_conv = "zero_conv" in args.checkpoint_dir
    # query_array = ["a mushroom"]
    for init_name in ["og_init","init_1","init_2","init_3"]:
        if init_name in args.checkpoint_dir:
            args.init = init_name
            break
    iter=5000
    net, flow, clip_model = get_networks(args.checkpoint_dir, args.init, zero_conv,args,iter=iter)
    for thresh in [0.05]:
        args.threshold = thresh
        voxels,raw_voxels = gen_voxels(query_array, net, flow,clip_model, args.checkpoint_dir)
        if not os.path.exists("voxels_%s/"%iter+args.output_dir):
            os.makedirs("voxels_%s/" % iter +args.output_dir)
        for i in range(voxels.shape[0]):
            print('voxel out path:',args.output_dir)
            np.save("voxels_%s/"%iter+args.output_dir+"/%s_%s.npy" % (query_array[i],thresh),voxels[i])

        save_voxels(voxels,args.output_dir,query_array,thresh)