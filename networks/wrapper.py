from utils import visualization
import torch
import torch.nn as nn

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
        
    noise = torch.Tensor(batch_size, args.emb_dims).normal_().to(args.device)
    decoder_embs = latent_flow_model.sample(text_features.device,batch_size, noise=noise, cond_inputs=text_features)
    out_3d = autoencoder(decoder_embs, query_points).view(batch_size, voxel_size, voxel_size, voxel_size)
    return out_3d