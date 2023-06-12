import clip
from torch.utils.tensorboard import SummaryWriter
from networks import autoencoder, latent_flows
from networks.autoencoder import ZeroConvDecoder
from train_autoencoder import parsing
import torch
import os
import random
from utils import helper
import json
import numpy as np

prompts_prefix_pool = ["a photo of a ", "a "]


def plt_render(out_3d_hard,iteration):
    # code for saving the binary voxel image renders
    # voxel_save is a function that takes in a voxel tensor and saves its rendering as a png
    # we save the renderings and then load them back in as tensors to display in tensorboard
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    for shape in range(min(num_shapes,3)):
        save_path = '/scratch/km3888/queries/%s/sample_%s_%s.png' % (args.id,iteration,shape)
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
    tensorboard_comment = 'q=%s_lr=%s_beta=%s_gpu=%s_baseline=%s_v=%s_k=%s_r=%s_s=%s'% (args.query_array,args.learning_rate,args.beta,args.gpu[0],args.uninitialized,args.num_voxels,args.num_views,args.renderer,args.seed)
    if args.use_zero_conv:
        tensorboard_comment += '_zero_conv'
    if args.contrast_lambda > 0:
        tensorboard_comment += '_c=%s' % args.contrast_lambda
    if args.temp!=1:
        tensorboard_comment += '_temp=%s' % args.temp
    if args.std_coeff>0:
        tensorboard_comment += '_std=%s' % args.std_coeff
    if args.slurm_id is not None:
        tensorboard_comment = str(args.slurm_id) + "/" + tensorboard_comment
    log_dir = '/scratch/km3888/clip_forge_runs/' + tensorboard_comment
    assert args.renderer in ['ea','nvr+']
    return SummaryWriter(log_dir=log_dir)

def get_networks(args):
    net = autoencoder.EncoderWrapper(args).to(args.device)    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, args.device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    if args.use_zero_conv:
        net.encoder.decoder = autoencoder.ZeroConvDecoder(net.encoder.decoder)
        net = net.to(args.device)
    if not args.uninitialized:
        checkpoint_nf_path = os.path.join(args.init_base + "/" + "models/prior/best.pt")
        checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
        latent_flow_network.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(args.init_base +"/"+ "models/autoencoder/best_iou.pt", map_location=args.device)
        net.load_state_dict(checkpoint['model'])
        net.eval()
    return net, latent_flow_network

def get_local_parser(mode="args"):
    parser = parsing(mode="parser")
    parser.add_argument("--num_blocks", type=int, default=5, help='Num of blocks for prior')
    parser.add_argument("--flow_type", type=str, default='realnvp_half', help='flow type: mf, glow, realnvp ')
    parser.add_argument("--num_hidden", type=int, default=1024, help='Number of parameter for flow model')
    parser.add_argument("--latent_load_checkpoint", type=str, default=None, help='Checkpoint to load latent flow model')
    parser.add_argument("--num_views",  type=int, default=5, metavar='N', help='Number of views')
    parser.add_argument("--clip_model_type",  type=str, default='B-32', metavar='N', help='what model to use')
    parser.add_argument("--noise",  type=str, default='add', metavar='N', help='add or remove')
    parser.add_argument("--seed_nf",  type=int, default=1, metavar='N', help='add or remove')
    parser.add_argument("--images_type",  type=str, default=None, help='img_choy13 or img_custom')
    parser.add_argument("--n_px",  type=int, default=224, help='Resolution of the image')
    parser.add_argument("--text_query",  type=str, default="")
    parser.add_argument("--beta",  type=float, default=75, help='regularization coefficient')
    parser.add_argument("--learning_rate",  type=float, default=01e-06, help='learning rate') #careful, base parser has "lr" param with different default value
    parser.add_argument("--use_tensorboard",  type=bool, default=True, help='use tensorboard')
    parser.add_argument("--query_array",  type=str, default=None, help='multiple queries') 
    parser.add_argument("--uninitialized",  type=bool, default=False, help='Use untrained networks')
    parser.add_argument("--num_voxels",  type=int, default=32, help='number of voxels')
    # parser.add_argument("--threshold",  type=float, default=0.5, help='threshold for voxelization')
    parser.add_argument("--renderer",  type=str, default='ea')
    parser.add_argument("--init_base",  type=str, default=" ", help='where is the initialization')
    parser.add_argument("--setting", type=int, default=None)
    parser.add_argument("--slurm_id", type=int, default=None)
    
    #checkpoint for nvr_renderer
    parser.add_argument("--nvr_renderer_checkpoint", type=str, default="/scratch/km3888/weights/nvr_plus.pt")
    parser.add_argument("--query_dir", type=str, default="/scratch/mp5847/queries")
    parser.add_argument("--contrast_lambda",type=float,default=0.1)
    parser.add_argument("--improved_contrast",action="store_true",help="improved contrast")
    parser.add_argument("--temp",type=float,default=1)
    
    parser.add_argument("--use_zero_conv", action="store_true", help="Use zero conv")
    parser.add_argument("--radius",type=float,default=0.75,help="radius for sphere prior")
    parser.add_argument("--background",type=str,default="default",help="background color")
    parser.add_argument("--std_coeff",type=float,default=0.0)
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
    helper.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_networks(args,iteration,wrapper):
    torch.save(wrapper.module.latent_flow_model.state_dict(), '%s/%s/flow_model_%s.pt' % (args.query_dir,args.id,iteration))
    torch.save(wrapper.module.autoencoder.state_dict(), '%s/%s/aencoder_%s.pt' % (args.query_dir,args.id,iteration))