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

from scripts.renderer import renderer_dict
import scripts.renderer.transform as dt

import torchvision
from torchvision.utils import save_image
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

import PIL

writer = SummaryWriter()

def clip_loss(query,args,clip_model,autoencoder,latent_flow_model,renderer,rotation,resizer,iter):
    text_emb,ims,hard_ims =generate_on_train_query(args,clip_model,autoencoder,latent_flow_model,renderer,query,32,rotation,resizer,iter)
    
    im_embs=clip_model.encode_image(ims)
    losses=-1*torch.cosine_similarity(text_emb,im_embs)
    loss = losses.mean()
    
    im_samples = [ims[0],ims[1],ims[2]]
    im_samples = [im.detach().cpu() for im in im_samples]
    
    hard_im_embs=clip_model.encode_image(hard_ims)
    hard_losses = -1*torch.cosine_similarity(text_emb,hard_im_embs)
    hard_loss = hard_losses.mean()
    
    return losses.mean(), hard_loss, im_samples

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

def generate_on_train_query(args,clip_model,autoencoder,latent_flow_model,renderer,text_in,batch_size,rotation,resizer,iter):
    transform = rotation 
    autoencoder.eval()
    latent_flow_model.eval()
    voxel_size = 32
    shape = (voxel_size, voxel_size, voxel_size)
    p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
    query_points = p.expand(batch_size, *p.size())
    
    text = clip.tokenize([text_in]).to(args.device)
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    noise = torch.Tensor(batch_size, args.emb_dims).normal_().to(args.device)
    decoder_embs = latent_flow_model.sample(batch_size, noise=noise, cond_inputs=text_features.repeat(batch_size,1))

    out_3d = autoencoder.decoding(decoder_embs, query_points).view(batch_size, voxel_size, voxel_size, voxel_size).to(args.device)
    out_3d_soft = torch.sigmoid(100*(out_3d-args.threshold))
    
    out_3d_hard = out_3d.detach() > args.threshold
    
    if not iter%20:
        voxel_render_loss = -1* evaluate_true_voxel(out_3d[0],args,clip_model,text_features,iter,text_in)
        writer.add_scalar('Loss/voxel_render_loss', voxel_render_loss, iter)
    # out_3d_soft = out_3d_soft.unsqueeze(0)
    resize_transform = resizer
    rgbs = []
    rgbs_hard = []
    for i in range(batch_size):
        rand_rotate_i = transform.rotate_random(out_3d_soft[i].float().unsqueeze(0).unsqueeze(0))
        rand_rotate_i_hard = transform.rotate_random(out_3d_hard[i].float().unsqueeze(0).unsqueeze(0))
        for axis in range(3):
            out_2d_i = renderer.render(volume=rand_rotate_i.squeeze(),axis=axis).double()
            out_2d_i_hard = renderer.render(volume=rand_rotate_i_hard.squeeze(),axis=axis).double()
            
            rgb_i = resize_transform(out_2d_i.unsqueeze(0).unsqueeze(0).expand(-1,3,-1,-1))
            rgb_i_hard = resize_transform(out_2d_i_hard.unsqueeze(0).unsqueeze(0).expand(-1,3,-1,-1))
        rgbs.append(rgb_i)
        rgbs_hard.append(rgb_i_hard)
    
    rgbs_hard = torch.cat(rgbs_hard,0)
    rgbs = torch.cat(rgbs,0)
    
    # clip_embs = clip_model.encode_image(rgbs)
            
    # save_image(out_2d_1, "rotated_car_out_2d_1.png")
    
    return text_features,rgbs,rgbs_hard
    

def evaluate_true_voxel(out_3d,args,clip_model,text_features,i,text):
    # code for saving the "true" voxel image, not useful right now
    out_3d_hard = out_3d>args.threshold
    voxel_save(out_3d_hard.squeeze().cpu().detach(), None, out_file='%s/sample_%s.png' % (text,i))
    # load the image that was saved and transform it to a tensor
    voxel_im = PIL.Image.open('%s/sample_%s.png' % (text,i)).convert('RGB')
    voxel_tensor = T.ToTensor()(voxel_im)
    writer.add_image('voxel image', voxel_tensor, i)
    # #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_tensor).unsqueeze(0)
    # get CLIP embedding
    voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
    voxel_similarity = torch.cosine_similarity(text_features, voxel_image_embedding)
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
    parser.add_argument("--beta",  type=float, default=100.0, help='regularization coefficient')
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

def test_train(args,clip_model,autoencoder,latent_flow_model,renderer):
    sample_query = args.text_query
    
    assert len(sample_query)
    rotation = dt.Transform(args.device)
    resizer = T.Resize(224)
    flow_optimizer=optim.Adam(latent_flow_model.parameters(), lr=01e-06)
    net_optimizer=optim.Adam(autoencoder.parameters(), lr=01e-06)
    
    losses = []
    
    # make directory for saving images with name of the text query using os.makedirs
    if not os.path.exists(sample_query):
        os.makedirs(sample_query)

    for iter in range(5000):
        flow_optimizer.zero_grad()
        net_optimizer.zero_grad()
        
        loss, hard_loss, rgb_images =clip_loss(sample_query,args,clip_model,autoencoder,latent_flow_model,renderer,rotation,resizer,iter)
        #save image
        grid = torchvision.utils.make_grid(rgb_images)
        writer.add_image('rendered image', grid, iter)
        
        loss.backward()
                
        losses.append(loss.item())
        
        writer.add_scalar('Loss/train', loss.item(), iter)
        writer.add_scalar('Loss/hard', hard_loss.item(), iter)
        
        flow_optimizer.step()
        net_optimizer.step()
    
    print(losses)
            
def main():
    args=get_local_parser()
    
    device, gpu_array = helper.get_device(args)
    args.device = device
    
    args, clip_model = get_clip_model(args)
    
    net = autoencoder.get_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    
    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
    checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
    latent_flow_network.load_state_dict(checkpoint['model'])
    
    param_dict={'device':args.device,'cube_len':32}
    renderer=renderer_dict['absorption_only'](param=param_dict)
    
    test_train(args,clip_model,net,latent_flow_network,renderer)
    
    print('xx')

if __name__=="__main__":
    main()