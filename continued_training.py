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

from torchvision.utils import save_image
import torchvision.transforms as T

import PIL

def clip_loss(query,args,clip_model,autoencoder,latent_flow_model,renderer):
    text_emb,im_1,im_2,im_3=generate_on_train_query(args,clip_model,autoencoder,latent_flow_model,renderer,query)
    images=torch.cat((im_1,im_2,im_3),0)
    im_embs=clip_model.encode_image(images)
    losses=-1*torch.cosine_similarity(text_emb,im_embs)
    return losses.mean()

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

def generate_on_train_query(args,clip_model,autoencoder,latent_flow_model,renderer,text_in,batch_size):
    transform = dt.Transform(args.device)
    autoencoder.eval()
    latent_flow_model.eval()
    voxel_size = 64
    shape = (voxel_size, voxel_size, voxel_size)
    p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
    query_points = p.expand(1, *p.size())
    
    text = clip.tokenize([text_in]).to(args.device)
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    noise = torch.Tensor(1, args.emb_dims).normal_().to(args.device)
    decoder_embs = latent_flow_model.sample(1, noise=noise, cond_inputs=text_features.repeat(1,1))

    out_3d = autoencoder.decoding(decoder_embs, query_points).view(1, voxel_size, voxel_size, voxel_size).to(args.device)
    out_3d_hard = out_3d>args.threshold
    out_3d_soft = torch.sigmoid(100*(out_3d-args.threshold))

    voxel_save(out_3d_hard.squeeze().cpu().detach(), None, out_file='plane_voxel_file.png')

    #load the image that was saved and transform it to a tensor
    voxel_im = PIL.Image.open('plane_voxel_file.png').convert('RGB')
    voxel_tensor = T.ToTensor()(voxel_im)
    #convert to 224x224 image with 3 channels
    voxel_tensor = T.Resize((224,224))(voxel_tensor).unsqueeze(0)
    #get CLIP embedding
    voxel_image_embedding = clip_model.encode_image(voxel_tensor.to(args.device))
    voxel_similarity = torch.cosine_similarity(text_features, voxel_image_embedding)
    
    out_3d_soft = out_3d_soft.unsqueeze(0)
    out_3d_soft = transform.rotate_random(out_3d_soft.float()).squeeze(0)

    out_2d_1 = renderer.render(volume=out_3d_soft,axis=1).double()
    out_2d_2 = renderer.render(volume=out_3d_soft,axis=2).double()
    out_2d_3 = renderer.render(volume=out_3d_soft,axis=3).double()
    
    transform = T.Resize(224)
    
    rgb_1 = transform(out_2d_1.unsqueeze(1).expand(-1,3,-1,-1))
    rgb_2 = transform(out_2d_2.unsqueeze(1).expand(-1,3,-1,-1))
    rgb_3 = transform(out_2d_3.unsqueeze(1).expand(-1,3,-1,-1))
    
    out_2d_1_emb = clip_model.encode_image(rgb_1)
    out_2d_2_emb = clip_model.encode_image(rgb_2)
    out_2d_3_emb = clip_model.encode_image(rgb_3)
    
    def compute_sims(text_arg):
        texty = clip.tokenize([text_arg]).to(args.device)
        texty_features = clip_model.encode_text(texty)
        texty_features = texty_features / texty_features.norm(dim=-1, keepdim=True)
        
        sim_1 = torch.cosine_similarity(texty_features, out_2d_1_emb)
        sim_2 = torch.cosine_similarity(texty_features, out_2d_2_emb)
        sim_3 = torch.cosine_similarity(texty_features, out_2d_3_emb)
        
        return [x.item() for x in [sim_1,sim_2,sim_3]]
    
    sim_1 = torch.cosine_similarity(text_features, out_2d_1_emb)
    sim_2 = torch.cosine_similarity(text_features, out_2d_2_emb)
    sim_3 = torch.cosine_similarity(text_features, out_2d_3_emb)
    
    save_image(out_2d_1, "rotated_car_out_2d_1.png")
    save_image(out_2d_2, "rotated_car_out_2d_2.png")
    save_image(out_2d_3, "rotated_car_out_2d_3.png")
    
    return text_features,rgb_1,rgb_2,rgb_3
    

def generate_on_query_text(args, clip_model, autoencoder, latent_flow_model):
    autoencoder.eval()
    latent_flow_model.eval()
    clip_model.eval()
    save_loc = args.generate_dir + "/"  
    count = 1
    num_figs = 3
    with torch.no_grad():
        voxel_size = 64
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
       
        for text_in in args.text_query:
            text = clip.tokenize([text_in]).to(args.device)
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            noise = torch.Tensor(num_figs, args.emb_dims).normal_().to(args.device)
            noise = torch.zeros_like(noise)
            
            decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))

            out = autoencoder.decoding(decoder_embs, query_points)
            
            if args.output_type == "Implicit":
                voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
                visualization.multiple_plot_voxel(voxels_out, save_loc=save_loc +"{}_text_query.png".format(text_in))
            elif args.output_type == "Pointcloud":
                pred = out.detach().cpu().numpy()
                visualization.multiple_plot(pred,  save_loc=save_loc +"{}_text_query.png".format(text_in))
                
    latent_flow_model.train()

def train_one_epoch(args, latent_flow_model, train_dataloader, optimizer, epoch):
    loss_prob_array = []
    loss_array = []
    latent_flow_model.train()
    
    for data in train_dataloader:
        optimizer.zero_grad()
        train_embs, train_cond_embs = data
        train_embs = train_embs.type(torch.FloatTensor).to(args.device)
        train_cond_embs = train_cond_embs.type(torch.FloatTensor).to(args.device)
        
        if args.noise == "add":
            train_embs = train_embs + 0.1 * torch.randn(train_embs.size(0), args.emb_dims).to(args.device)
        
        loss_log_prob = - latent_flow_model.log_prob(train_embs, train_cond_embs).mean()  
        loss = loss_log_prob
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())
        loss_prob_array.append(loss_log_prob.item())
    loss_array = np.asarray(loss_array)
    loss_prob_array = np.asarray(loss_prob_array)
    logging.info("[Train] Epoch {} Train loss {} Prob loss {} ".format(epoch, np.mean(loss_array), np.mean(loss_prob_array))) 

def get_local_parser(mode="args"):
    parser = parsing(mode="parser")
    parser.add_argument("--num_blocks", type=int, default=5, help='Num of blocks for prior')
    parser.add_argument("--flow_type", type=str, default='realnvp_half', help='flow type: mf, glow, realnvp ')
    parser.add_argument("--num_hidden", type=int, default=1024, help='Number of parameter for flow model')
    parser.add_argument("--latent_load_checkpoint", type=str, default=None, help='Checkpoint to load latent flow model')
    parser.add_argument("--text_query", nargs='+', default=None, metavar='N', help='text query array')
    parser.add_argument("--num_views",  type=int, default=5, metavar='N', help='Number of views')
    parser.add_argument("--clip_model_type",  type=str, default='B-32', metavar='N', help='what model to use')
    parser.add_argument("--noise",  type=str, default='add', metavar='N', help='add or remove')
    parser.add_argument("--seed_nf",  type=int, default=1, metavar='N', help='add or remove')
    parser.add_argument("--images_type",  type=str, default=None, help='img_choy13 or img_custom')
    parser.add_argument("--n_px",  type=int, default=224, help='Resolution of the image')
    parser.add_argument("--checkpoint_dir_base", type=str, default=None)
    parser.add_argument("--checkpoint_dir_prior", type=str, default=None)
    parser.add_argument("--checkpoint_nf",  type=str, default="best", help='what is the checkpoint for nf')
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

def test_train(args,clip_model,autoencoder,latent_flow_model,renderer):
    sample_query = 'an airplane'
    flow_optimizer=optim.Adam(latent_flow_model.parameters(), lr=01e-06)
    net_optimizer=optim.Adam(autoencoder.parameters(), lr=01e-06)
    
    loss=clip_loss(sample_query,args,clip_model,autoencoder,latent_flow_model,renderer)
    loss.backward()
    
    flow_optimizer.step()
    net_optimizer.step()
    
    new_loss=clip_loss(sample_query,args,clip_model,autoencoder,latent_flow_model,renderer)
    pass
    
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
    
    param_dict={'device':args.device,'cube_len':64}
    renderer=renderer_dict['absorption_only'](param=param_dict)
    
    test_train(args,clip_model,net,latent_flow_network,renderer)
    
    print('xx')

if __name__=="__main__":
    main()