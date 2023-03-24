import clip
from torch.utils.tensorboard import SummaryWriter
from networks import autoencoder, latent_flows
from train_autoencoder import parsing
import torch
import sys
import os
import numpy as np
import random

query_arrays = {
                "wineglass": ["wineglass"],
                "spoon": ["spoon"],
                "fork": ["fork"],
                "hammer": ["hammer"],
                "three": ["spoon","fork","knife"],
                "four": ["spoon","fork","wineglass","knife"],
                "six": ['wineglass','spoon','fork','knife','screwdriver','hammer'],
                "nine": ['wineglass','spoon','fork','knife','screwdriver','hammer',"soccer ball", "football","plate"],
                "fourteen": ["wineglass','spoon','fork','knife','screwdriver','hammer","pencil","screw","plate","mushroom","umbrella","thimble","sombrero","sandal"]
}

prompts_prefix_pool = ["a photo of a ", "a "]

def get_prompts(obj, num_prompts):
    
    #get prompts for each object
    prompts = []
    
    for i in range(num_prompts):
        prompts.append(random.choice(prompts_prefix_pool) + obj)
    return prompts

def make_init_dict():
    og_init = {"ae_path":"models/autoencoder/best_iou.pt", "flow_path":"models/prior/best.pt","num_blocks":5,"num_hidden":1024,"emb_dim":128}
    init_1 = {"ae_path":"New__Autoencoder_Shapenet_Voxel_Implicit_128_add_noise_1/checkpoints/best_iou.pt", \
                "flow_path":"New__Autoencoder_Shapenet_Voxel_Implicit_128_add_noise_1/Clip_Conditioned_realnvp_half_8_best_iou_1_B-32_1024_1/checkpoints/best.pt",\
                "num_blocks":8,"num_hidden":1024,"emb_dim":128}
    init_2 = {"ae_path":"New__Autoencoder_Shapenet_Voxel_Implicit_256_add_noise_1/checkpoints/best_iou.pt", \
                "flow_path":"New__Autoencoder_Shapenet_Voxel_Implicit_256_add_noise_1/Clip_Conditioned_realnvp_half_8_best_iou_1_B-32_1024_1/checkpoints/best.pt",\
                "num_blocks":8,"num_hidden":1024,"emb_dim":256}
    init_3 = {"ae_path":"New__Autoencoder_Shapenet_Voxel_Implicit_256_add_noise_1/checkpoints/best_iou.pt", \
                "flow_path":"New__Autoencoder_Shapenet_Voxel_Implicit_256_add_noise_1/Clip_Conditioned_realnvp_half_8_best_iou_1_B-32_2048_1/checkpoints/best.pt",\
                "num_blocks":8,"num_hidden":2048,"emb_dim":256}
    
    init_dict = {"og_init":og_init,"init_1":init_1,"init_2":init_2,"init_3":init_3}
    return init_dict
    

def make_writer(args):
    if not args.use_tensorboard:
        return None
    tensorboard_comment = 'q=%s_lr=%s_beta=%s_gpu=%s_baseline=%s_v=%s_k=%s_r=%s'% (args.query_array,args.learning_rate,args.beta,args.gpu[0],args.uninitialized,args.num_voxels,args.num_views,args.renderer)
    tensorboard_comment += "_init=%s" % args.init
    if args.switch_point is not None:
        tensorboard_comment += '_s=%s' % args.switch_point
    if args.orthogonal:
        tensorboard_comment += '_orthogonal'
    if args.slurm_id is not None:
        tensorboard_comment = str(args.slurm_id) + "/" + tensorboard_comment
    tensorboard_comment += 'amp'
    log_dir = '/scratch/km3888/clip_forge_runs/' + tensorboard_comment
    assert args.renderer in ['ea','nvr+']
    return SummaryWriter(log_dir=log_dir)

def get_networks(args,init_dict):
    args.emb_dims = init_dict["emb_dim"]
    args.num_blocks = init_dict["num_blocks"]
    args.num_hidden = init_dict["num_hidden"]
    net = autoencoder.EncoderWrapper(args).to(args.device)    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, args.device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    if not args.uninitialized:
        sys.stdout.flush()
        checkpoint_nf_path = os.path.join(args.init_base + "/" + init_dict["flow_path"])
        checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
        latent_flow_network.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(args.init_base +"/"+ init_dict["ae_path"], map_location=args.device)
        net.load_state_dict(checkpoint['model'])
        net.eval()
        #calculate total parameters in autoencoder and latent flow
        total_params_ae = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total_params_nf = sum(p.numel() for p in latent_flow_network.parameters() if p.requires_grad)
        print("Total parameters in Autoencoder: %s" % total_params_ae)
        print("Total parameters in Latent Flow: %s" % total_params_nf)
        print("Total parameters in initialization: %s" % (total_params_ae + total_params_nf))
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
    parser.add_argument("--switch_point",type=float, default=None, help='switch point for the renderer')
    parser.add_argument("--renderer",  type=str, default='ea')
    parser.add_argument("--orthogonal",  type=bool, default=False, help='use orthogonal views')
    parser.add_argument("--init",  type=str, default="og_init", help='what is the initialization')
    parser.add_argument("--init_base",  type=str, default="/scratch/km3888/inits", help='where is the initialization')
    parser.add_argument("--setting", type=int, default=None)
    parser.add_argument("--slurm_id", type=int, default=None)
    
    #checkpoint for nvr_renderer
    parser.add_argument("--nvr_renderer_checkpoint", type=str, default="/scratch/mp5847/general_clip_forge/nvr_plus.pt")
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

def get_text_embeddings(args,clip_model,query_array):
    # get the text embedding for each query
    prompts = []
    for obj in set(query_array):
        prompts.extend(get_prompts(obj, args.num_views))
    print("in get text embeddings")
    print(query_array)
    text_tokens = []
    with torch.no_grad():
        for text in prompts:
            print("text is: ", text)
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

