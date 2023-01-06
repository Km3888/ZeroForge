import os
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

from rendering.nvr_server import NVR_Renderer
from rendering.baseline_renderer import BaselineRenderer

import torchvision
from torchvision.utils import save_image
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

import PIL



def clip_loss(args,query_array,clip_model,autoencoder,latent_flow_model,renderer,resizer,iter,text_features=None):
    # text_emb,ims = generate_single_query(args,clip_model,autoencoder,latent_flow_model,renderer,query,args.batch_size,rotation,resizer,iter)
    
    text_embs,ims = generate_for_query_array(args,clip_model,autoencoder,latent_flow_model,renderer,query_array,resizer,iter,text_features=text_features)
    
    im_embs=clip_model.encode_image(ims)
    text_embs=text_embs.unsqueeze(1).expand(-1,3,-1).reshape(-1,512)
    losses=-1*torch.cosine_similarity(text_embs,im_embs)
    loss = losses.mean()
    
    if args.use_tensorboard and not iter%10:
        im_samples= ims.view(-1,3,224,224)
        grid = torchvision.utils.make_grid(im_samples, nrow=3)
        writer.add_image('images', grid, iter)

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

def generate_for_query_array(args,clip_model,autoencoder,latent_flow_model,renderer,query_array,resizer,iter,text_features=None):
    clip_model.eval()
    autoencoder.train()
    latent_flow_model.eval() # has to be in .eval() mode for the sampling to work (which is bad but whatever)
    
    voxel_size = args.num_voxels
    batch_size = len(query_array)
        
    shape = (voxel_size, voxel_size, voxel_size)
    p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
    query_points = p.expand(batch_size, *p.size())
    
     # get the text embedding for each query
    if text_features is None:
        text_features = get_text_embeddings(args,clip_model,query_array)
        text_features = text_features.clone()
    #REFACTOR compute text_features outside this method
    
    noise = torch.Tensor(batch_size, args.emb_dims).normal_().to(args.device)
    decoder_embs = latent_flow_model.sample(batch_size, noise=noise, cond_inputs=text_features)
    
    out_3d = autoencoder.decoding(decoder_embs, query_points).view(batch_size, voxel_size, voxel_size, voxel_size).to(args.device)
    out_3d_soft = torch.sigmoid(args.beta*(out_3d-args.threshold)).clone()
   
    if not iter%50:
        out_3d_hard = out_3d.detach() > args.threshold
        #Currently only doing all 3 angles for ea, could try something similar
        #for nvr+ once I understand the camera angle better
        #renderer expects [batch,voxel_size,voxel_size,voxel_size]    
        rgbs_hard = renderer.render(out_3d_hard.float()).double().to('cuda:0')
        rgbs_hard = resizer(rgbs_hard)
        
        hard_im_embeddings = clip_model.encode_image(rgbs_hard)
        
        text_labels = text_features.unsqueeze(1).expand(-1,3,-1).reshape(-1,512)
        hard_loss = -1*torch.cosine_similarity(text_labels,hard_im_embeddings).mean()
        #write to tensorboard
        voxel_render_loss = -1* evaluate_true_voxel(out_3d_hard,args,clip_model,text_features,iter)
        if args.use_tensorboard:
            writer.add_scalar('Loss/hard_loss', hard_loss, iter)
            writer.add_scalar('Loss/voxel_render_loss', voxel_render_loss, iter)
    
    #REFACTOR put all these into a single method which works for hard or soft
    rgbs = renderer.render(volume=out_3d_soft).double()            
    rgbs = resizer(rgbs)
        
    return text_features,rgbs
    

def evaluate_true_voxel(out_3d,args,clip_model,text_features,i):
    # code for saving the "true" voxel image, not useful right now
    out_3d_hard = out_3d>args.threshold
    voxel_ims=[]
    num_shapes = out_3d_hard.shape[0]
    for shape in range(num_shapes):
        voxel_save(out_3d_hard[shape].squeeze().cpu().detach(), None, out_file='queries/%s/sample_%s_%s.png' % (args.query_array,i,shape))
        # load the image that was saved and transform it to a tensor
        voxel_im = PIL.Image.open('queries/%s/sample_%s_%s.png' % (args.query_array,i,shape)).convert('RGB')
        voxel_tensor = T.ToTensor()(voxel_im)
        voxel_ims.append(voxel_tensor.unsqueeze(0))
    
    voxel_ims = torch.cat(voxel_ims,0)
    grid = torchvision.utils.make_grid(voxel_ims, nrow=num_shapes)
    
    if args.use_tensorboard:
        writer.add_image('voxel image', grid, i)
    # #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_ims)
    # get CLIP embedding
    voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
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
    if not os.path.exists('queries/%s' % args.query_array):
        os.makedirs('queries/%s' % args.query_array)

    for iter in range(20000):
        flow_optimizer.zero_grad()
        net_optimizer.zero_grad()
        
        loss = clip_loss(args,query_array,clip_model,autoencoder,latent_flow_model,renderer,resizer,iter,text_features)        
        
        loss.backward()
                
        losses.append(loss.item())
        
        if args.use_tensorboard:
            writer.add_scalar('Loss/train', loss.item(), iter)
        
        flow_optimizer.step()
        net_optimizer.step()
    
    print(losses)
            
def main(args):
    device, gpu_array = helper.get_device(args)
    args.device = device
    
    args, clip_model = get_clip_model(args)
    
    net = autoencoder.get_model(args).to(args.device)
    if not args.uninitialized:
        checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
        net.load_state_dict(checkpoint['model'])
        net.eval()
    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    if not args.uninitialized:
        checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
        checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
        latent_flow_network.load_state_dict(checkpoint['model'])
    
    param_dict={'device':args.device,'cube_len':args.num_voxels}
    if args.renderer == 'ea':
        renderer=BaselineRenderer('absorption_only',param_dict)
    elif args.renderer == 'nvr+':
        renderer = NVR_Renderer()
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
    if args.use_tensorboard:
        writer=SummaryWriter(comment='_%s_lr=%s_beta=%s_gpu=%s_baseline=%s_v=%s'% (args.query_array,args.learning_rate,args.beta,args.gpu[0],args.uninitialized,args.num_voxels))
    assert args.renderer in ['ea','nvr+']
    main(args)