import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import os.path as osp
import logging

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

def get_type(visual_model):
    return visual_model.conv1.weight.dtype

def clip_loss(args,query_array,visual_model,autoencoder,latent_flow_model,renderer,resizer,iter,text_features):
    # text_emb,ims = generate_single_query(args,clip_model,autoencoder,latent_flow_model,renderer,query,args.batch_size,rotation,resizer,iter)
    out_3d = gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features)
    out_3d_soft = torch.sigmoid(args.beta*(out_3d-args.threshold))#.clone()
    
    #REFACTOR put all these into a single method which works for hard or soft
    ims = renderer.render(out_3d_soft).double()
    ims = resizer(ims)

    # im_embs=clip_model.encode_image(ims)
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

def get_text_embeddings(args,clip_model,query_array):
    # get the text embedding for each query
    text_tokens = []
    with torch.no_grad():
        for text in query_array:
            text_tokens.append(clip.tokenize([text]).detach().to(args.device))
        
    text_tokens = torch.cat(text_tokens,dim=0)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features):
    autoencoder.train()
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
    out_3d = gen_shapes(query_array,args,autoencoder,latent_flow_model,text_features)
    #save out_3d to numpy file
    # with open(f'out_3d/{args.learning_rate}_{args.query_array}/out_3d_{iter}.npy', 'wb') as f:
    #     np.save(f, out_3d.cpu().detach().numpy())
    
    out_3d_hard = out_3d.detach() > args.threshold
    rgbs_hard = renderer.render(out_3d_hard.float()).double().to(args.device)
    rgbs_hard = resizer(rgbs_hard)
    
    # hard_im_embeddings = clip_model.encode_image(rgbs_hard)
    hard_im_embeddings = visual_model(rgbs_hard.type(visual_model_type))
    
    if args.renderer=='ea':
        #baseline renderer gives 3 dimensions
        text_features=text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512)
    hard_loss = -1*torch.cosine_similarity(text_features,hard_im_embeddings).mean()
    #write to tensorboard
    voxel_render_loss = -1* evaluate_true_voxel(out_3d_hard,args,visual_model,text_features,iter)
    if args.use_tensorboard:
        args.writer.add_scalar('Loss/hard_loss', hard_loss, iter)
        args.writer.add_scalar('Loss/voxel_render_loss', voxel_render_loss, iter)

def evaluate_true_voxel(out_3d,args,visual_model,text_features,i):
    # code for saving the "true" voxel image
    out_3d_hard = out_3d>args.threshold
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    for shape in range(num_shapes):
        save_path = 'queries/%s/sample_%s_%s.png' % (args.writer.log_dir[5:],i,shape)
        voxel_save(out_3d_hard[shape].squeeze().detach().cpu(), None, out_file=save_path)
        # load the image that was saved and transform it to a tensor
        voxel_im = PIL.Image.open(save_path).convert('RGB')
        voxel_tensor = T.ToTensor()(voxel_im)
        voxel_ims.append(voxel_tensor.unsqueeze(0))
    
    voxel_ims = torch.cat(voxel_ims,0)
    grid = torchvision.utils.make_grid(voxel_ims, nrow=num_shapes)

    for shape in range(num_shapes):
        save_path = 'queries/%s/sample_%s_%s.png' % (args.writer.log_dir[5:],i,shape)
        os.remove(save_path)

    if args.use_tensorboard:
        args.writer.add_image('voxel image', grid, i)
    # #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_ims)
    # get CLIP embedding
    # voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
    voxel_image_embedding = visual_model(voxel_tensor.to(args.device).type(visual_model_type))
    voxel_similarity = torch.cosine_similarity(text_features, voxel_image_embedding).mean()
    return voxel_similarity


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
    parser.add_argument("--checkpoint_dir_base", type=str, default=None)
    parser.add_argument("--checkpoint_dir_prior", type=str, default=None)
    parser.add_argument("--checkpoint_nf",  type=str, default="best", help='what is the checkpoint for nf')
    parser.add_argument("--text_query",  type=str, default="")
    parser.add_argument("--beta",  type=float, default=75, help='regularization coefficient')
    parser.add_argument("--learning_rate",  type=float, default=01e-06, help='learning rate') #careful, base parser has "lr" param with different default value
    parser.add_argument("--use_tensorboard",  type=bool, default=True, help='use tensorboard')
    parser.add_argument("--query_array",  type=str, default=None, help='multiple queries') 
    parser.add_argument("--uninitialized",  type=bool, default=False, help='Use untrained networks')
    parser.add_argument("--num_voxels",  type=int, default=32, help='number of voxels')
    # parser.add_argument("--threshold",  type=float, default=0.5, help='threshold for voxelization')
    parser.add_argument("--renderer",  type=str, default='ea')
    parser.add_argument("--setting", type=int, default=None)
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

def test_train(args,clip_model,autoencoder,latent_flow_model,renderer):    
    resizer = T.Resize(224)
    flow_optimizer=optim.Adam(latent_flow_model.parameters(), lr=args.learning_rate)
    net_optimizer=optim.Adam(autoencoder.parameters(), lr=args.learning_rate)

    losses = []
    
    if args.query_array in query_arrays:
        query_array = query_arrays[args.query_array]
    else:
        query_array = [args.query_array]
    if len(query_array) ==1:
        query_array = query_array*args.num_views
    text_features = get_text_embeddings(args,clip_model,query_array).detach()
    # make directory for saving images with name of the text query using os.makedirs
    if not os.path.exists('queries/%s' % args.writer.log_dir[5:]):
        os.makedirs('queries/%s' % args.writer.log_dir[5:])

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
        
        if not iter%300:
            do_eval(renderer,query_array,args,visual_model,autoencoder,latent_flow_model,resizer,iter,text_features)
        
        if not (iter%5000) and iter!=0:
            #save encoder and latent flow network
            torch.save(latent_flow_model.state_dict(), 'queries/%s/flow_model_%s.pt' % (args.writer.log_dir[5:],iter))
            torat.save(autoencoder.encoder.state_dict(), 'queries/%s/aencoder_%s.pt' % (args.writer.log_dir[5:],iter))
            
        flow_optimizer.zero_grad()
        net_optimizer.zero_grad()
        
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
    torch.save(latent_flow_model.state_dict(), 'queries/%s/flow_model.pt' % args.writer.log_dir[5:])
    torch.save(autoencoder.encoder.state_dict(), 'queries/%s/final_aencoder.pt' % args.writer.log_dir[5:])
    
    print(losses)
            
def main(args):
    if args.use_tensorboard:
        args.writer=SummaryWriter(comment='_%s_lr=%s_beta=%s_gpu=%s_baseline=%s_v=%s_k=%s'% (args.query_array,args.learning_rate,args.beta,args.gpu[0],args.uninitialized,args.num_voxels,args.num_views))
    assert args.renderer in ['ea','nvr+']
    
    # if not os.path.exists(f'out_3d/{args.learning_rate}_{args.query_array}'):
    #     os.mkdir(f'out_3d/{args.learning_rate}_{args.query_array}')

    device, gpu_array = helper.get_device(args)
    args.device = device
    
    print("Using device: ", device)

    args, clip_model = get_clip_model(args)
    
    net = autoencoder.EncoderWrapper(args).to(args.device)

    if not args.uninitialized:
        checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
        net.load_state_dict(checkpoint['model'])
        net.eval()
    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    if not args.uninitialized:
        print(args.checkpoint_dir_prior)
        print(args.checkpoint)
        sys.stdout.flush()
        checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
        checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
        latent_flow_network.load_state_dict(checkpoint['model'])
    
    param_dict={'device':args.device,'cube_len':args.num_voxels}
    if args.renderer == 'ea':
        renderer=BaselineRenderer('absorption_only',param_dict)
    elif args.renderer == 'nvr+':
        renderer = NVR_Renderer()
        renderer.model.to(args.device)
    net = nn.DataParallel(net)

    test_train(args,clip_model,net,latent_flow_network,renderer)
    
query_arrays = {
                "wineglass": ["wineglass"],
                "spoon": ["spoon"],
                "fork": ["fork"],
                "hammer": ["hammer"],
                "six": ['wineglass','spoon','fork','knife','screwdriver','hammer'],
                "nine": ['wineglass','spoon','fork','knife','screwdriver','hammer',"soccer ball", "football","plate"],
                "fourteen": ["wineglass','spoon','fork','knife','screwdriver','hammer","pencil","screw","screwdriver","plate","mushroom","umbrella","thimble","sombrero","sandal"]
}
#REFACTOR put query arrays in a separate file

if __name__=="__main__":
    args=get_local_parser()
    print('renderer %s' % args.renderer)
    import sys; sys.stdout.flush()
    main(args)
