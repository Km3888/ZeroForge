import clip
from torch.utils.tensorboard import SummaryWriter
from networks import autoencoder, latent_flows
from networks.autoencoder import ZeroConvDecoder
import torch
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import PIL
import torchvision.transforms as T

prompts_prefix_pool = ["a photo of a ", "a "]


def plt_render(out_3d_hard,iteration,args):
    # code for saving the binary voxel image renders
    # voxel_save is a function that takes in a voxel tensor and saves its rendering as a png
    # we save the renderings and then load them back in as tensors to display in tensorboard
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    for shape in range(min(num_shapes,3)):
        save_path = '%s/%s/sample_%s_%s.png' % (args.log_dir,args.id,iteration,shape)
        voxel_save(out_3d_hard[shape].squeeze().detach().cpu(), None, out_file=save_path)
        # load the image that was saved and transform it to a tensor
        voxel_im = PIL.Image.open(save_path).convert('RGB')
        voxel_tensor = T.ToTensor()(voxel_im)
        voxel_ims.append(voxel_tensor.unsqueeze(0))
    
    voxel_ims = torch.cat(voxel_ims,0)

    return voxel_ims

def get_query_array(args):
    with open("query_arrays.json", "r") as json_file:
        query_arrays = json.load(json_file)
    if args.query_array in query_arrays:
        query_array = query_arrays[args.query_array]
    else:
        query_array = [args.query_array]
    query_array = query_array*args.num_views
    args.unique = len(set(query_array))
    return query_array

def get_prompts(obj, num_prompts):
    #get prompts for each object
    prompts = []
    for i in range(num_prompts):
        prompts.append(random.choice(prompts_prefix_pool) + obj)
    return prompts    

def make_writer(args):
    if not args.use_tensorboard:
        return None
    tensorboard_comment = 'q=%s_lr=%s_beta=%s_gpu=%s_baseline=%s_k=%s_r=%s_s=%s'% (args.query_array,args.learning_rate,args.beta,args.gpu[0],args.uninitialized,args.num_views,args.renderer,args.seed)
    if args.use_zero_conv:
        tensorboard_comment += '_zero_conv'
    if args.contrast_lambda > 0:
        tensorboard_comment += '_c=%s' % args.contrast_lambda
    if args.temp!=1:
        tensorboard_comment += '_temp=%s' % args.temp
    if args.slurm_id is not None:
        tensorboard_comment = str(args.slurm_id) + "/" + tensorboard_comment
    log_dir = args.log_dir + tensorboard_comment
    assert args.renderer in ['ea','nvr+']
    return SummaryWriter(log_dir=log_dir)

def get_networks(args):
    net = autoencoder.EncoderWrapper(args).to(args.device)    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, args.device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    if args.use_zero_conv:
        net.encoder.decoder = autoencoder.ZeroConvDecoder(net.encoder.decoder)
        net = net.to(args.device)
    if not args.uninitialized:
        checkpoint_nf_path = os.path.join(args.init_base, "prior/best.pt")
        checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
        latent_flow_network.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(os.path.join(args.init_base,"autoencoder/best_iou.pt"), map_location=args.device)
        net.load_state_dict(checkpoint['model'])
        net.eval()
    return net, latent_flow_network

def get_local_parser(mode="args"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_type", type=str, default='Voxel', help='What is the input representation')
    parser.add_argument("--output_type", type=str, default='Implicit', help='What is the output representation')
    parser.add_argument("--encoder_type", type=str, default='Voxel_Encoder_BN', help='what is the encoder')
    parser.add_argument("--decoder_type", type=str, default='Occ_Simple_Decoder', help='what is the decoder')
    parser.add_argument('--emb_dims', type=int, default=128, help='Dimension of embedding')
    parser.add_argument('--last_feature_transform', type=str, default="add_noise", help='add_noise or none')
    parser.add_argument('--reconstruct_loss_type', type=str, default="sum", help='bce or sum (mse) or mean (mse)')
    parser.add_argument('--pc_dims', type=int, default=1024, help='Dimension of embedding')
                            
    ### training details
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    parser.add_argument('--gpu', nargs='+' , default="0", help='GPU list')
    parser.add_argument('--optimizer', type=str, choices=('SGD', 'Adam'), default='Adam')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for voxel stuff')

    parser.add_argument("--num_blocks", type=int, default=5, help='Num of blocks for prior')
    parser.add_argument("--flow_type", type=str, default='realnvp_half', help='flow type: mf, glow, realnvp ')
    parser.add_argument("--num_hidden", type=int, default=1024, help='Number of parameter for flow model')
    parser.add_argument("--latent_load_checkpoint", type=str, default=None, help='Checkpoint to load latent flow model')
    parser.add_argument("--num_views",  type=int, default=3, metavar='N', help='Number of views')
    parser.add_argument("--clip_model_type",  type=str, default='B-32', metavar='N', help='what model to use')
    parser.add_argument("--noise",  type=str, default='add', metavar='N', help='add or remove')
    parser.add_argument("--seed_nf",  type=int, default=1, metavar='N', help='add or remove')
    parser.add_argument("--beta",  type=float, default=150, help='regularization coefficient')
    parser.add_argument("--learning_rate",  type=float, default=01e-05, help='learning rate') #careful, base parser has "lr" param with different default value
    parser.add_argument("--use_tensorboard",  type=bool, default=True, help='use tensorboard')
    parser.add_argument("--query_array",  type=str, default=None, help='multiple queries') 
    parser.add_argument("--uninitialized",  type=bool, default=False, help='Use untrained networks')
    parser.add_argument("--num_voxels",  type=int, default=128, help='number of voxels')
    parser.add_argument("--renderer",  type=str, default='nvr+')
    parser.add_argument("--init_base",  type=str, default="exps/models/", help='where is the initialization')
    parser.add_argument("--setting", type=int, default=None) #### Used for jobs run with SLURM
    parser.add_argument("--slurm_id", type=int, default=None) #### Used for jobs run with SLURM
    parser.add_argument("--log_dir", type=str, default="logs/")
    
    #checkpoint for nvr_renderer
    parser.add_argument("--nvr_renderer_checkpoint", type=str)
    parser.add_argument("--contrast_lambda",type=float,default=0.1)
    parser.add_argument("--improved_contrast",action="store_true",help="improved contrast")
    parser.add_argument("--temp",type=float,default=50)
    
    parser.add_argument("--use_zero_conv", action="store_true", help="Use zero conv")
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

def get_text_embeddings(args,clip_model,query_array):
    # get the text embedding for each query
    prompts = []
    for obj in query_array:
        prompts.extend(get_prompts(obj, 1))
    
    text_tokens = []
    with torch.no_grad():
        for text in prompts:
            text_tokens.append(clip.tokenize([text]).detach().to(args.device))
    
    text_tokens = torch.cat(text_tokens,dim=0)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def get_type(visual_model):
    return visual_model.conv1.weight.dtype

def get_clip_model(args):
    if args.clip_model_type == "B-16":
        print("Bigger model is being used B-16")
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=args.device)
        cond_emb_dim = 512
    elif args.clip_model_type == "RN50x16":
        print("Using the RN50x16 model")
        clip_model, clip_preprocess = clip.load("RN50x16", device=args.device)
        cond_emb_dim = 768
    else:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
        cond_emb_dim = 512
    
    input_resolution = clip_model.visual.input_resolution
    #train_cond_embs_length = clip_model.train_cond_embs_length
    vocab_size = clip_model.vocab_size
    #cond_emb_dim  = clip_model.embed_dim
    #print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("cond_emb_dim:", cond_emb_dim)
    print("Input resolution:", input_resolution)
    #print("train_cond_embs length:", train_cond_embs_length)
    print("Vocab size:", vocab_size)
    args.n_px = input_resolution
    args.cond_emb_dim = cond_emb_dim
    return args, clip_model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_networks(args,iteration,wrapper):
    torch.save(wrapper.module.latent_flow_model.state_dict(), '%s/%s/flow_model_%s.pt' % (args.log_dir,args.id,iteration))
    torch.save(wrapper.module.autoencoder.state_dict(), '%s/%s/aencoder_%s.pt' % (args.log_dir,args.id,iteration))


def voxel_save(voxels, text_name, out_file=None, transpose=True, show=False):

    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    #fig = plt.figure()
    fig = plt.figure(figsize=(40,20))
    
    ax = fig.add_subplot(111, projection=Axes3D.name)
    if transpose == True:
        voxels = voxels.transpose(2, 0, 1)
    #else:
        #voxels = voxels.transpose(2, 0, 1)
    

    ax.voxels(voxels, edgecolor='k', facecolors='coral', linewidth=0.5)
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    # Hide grid lines
    plt.grid(False)
    plt.axis('off')
    
    if text_name != None:
        plt.title(text_name, {'fontsize':30}, y=0.15)
    #plt.text(15, -0.01, "Correlation Graph between Citation & Favorite Count")

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.axis('off')
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def get_device(args):
    gpu_string = "cuda:"
    gpu_array = []
    length = 1
    for i in args.gpu:
        if length == len(args.gpu):
            gpu_string = gpu_string + i
        else:
            gpu_string = gpu_string + i + ","
        gpu_array.append(int(i))
        length = length + 1
    return gpu_string, gpu_array

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p
