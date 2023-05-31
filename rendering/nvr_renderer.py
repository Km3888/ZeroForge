import torch
import torch.nn as nn

from rendering.nvr_torch.torch_model import NVR_Plus
from rendering.preprocess import diff_preprocess,Preprocessor
import numpy as np

class NVR_Renderer(nn.Module):
    
    def __init__(self, args, device):
        super(NVR_Renderer, self).__init__()
        self.model = NVR_Plus()
        self.model.load_state_dict(torch.load(args.nvr_renderer_checkpoint, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
    
    def forward(self,voxels):
        light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773]).astype(np.float32)
        light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)
        light_position = torch.from_numpy(light_position).to(voxels.device)
        light_position = torch.cat([light_position]*voxels.shape[0],dim=0)
        
        #count how many nan entries are in light_position

        batch_size=voxels.shape[0]
        rotation_angles = self.generate_random_rotations(batch_size).astype(np.float32)

        final_composite,interpolated_voxels = diff_preprocess(voxels,rotation_angles)
        final_composite = (final_composite - 0.5)*2
        interpolated_voxels = interpolated_voxels.permute(0,4,1,2,3)

        # # self.model = self.model.to(voxels.device)
        output=self.model(interpolated_voxels,final_composite,light_position)
        return output*0.5+0.5

    def generate_random_rotations(self,batch_size):
        # generate a random point on the unit sphere
        u = np.random.uniform(low=0.0,high=1.0,size=(batch_size, 1))
        v = np.random.uniform(low=0.0,high=1.0,size=(batch_size, 1))

        theta = np.arccos(2*u-1)
        phi = 2*np.pi*v
        rotation_z = np.zeros((batch_size,1))

        rotation_angles = np.concatenate([theta,phi,rotation_z],axis=1)
        return rotation_angles
