import torch

from rendering.nvr_torch.torch_model import NVR_Plus
from rendering.preprocess import diff_preprocess
import numpy as np

class NVR_Renderer:
    
    def __init__(self):
        self.model = NVR_Plus()
        self.model.load_state_dict(torch.load('/scratch/mp5847/general_clip_forge/nvr_plus.pt'))
        self.model.to('cuda:0') #TODO make device arbitrary
        self.model.eval()
    
    def render(self,voxels):
        light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773]).astype(np.float32)
        light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)
        light_position = torch.from_numpy(light_position).to(voxels.device)
        light_position = torch.cat([light_position]*voxels.shape[0],dim=0)

        batch_size=voxels.shape[0]
        rotation_x = -np.deg2rad(np.random.uniform(0,360,size=(batch_size,1))).astype(np.float32)
        rotation_y = -np.deg2rad(np.random.uniform(0,360,size=(batch_size,1))).astype(np.float32)
        rotation_z = -np.deg2rad(np.random.uniform(0,360,size=(batch_size,1))).astype(np.float32)
        rotation_angles = np.concatenate([rotation_x,rotation_y,rotation_z],axis=1)
        
        final_composite,interpolated_voxels = diff_preprocess(voxels,rotation_angles)
        final_composite = (final_composite - 0.5)*2
        interpolated_voxels = interpolated_voxels.permute(0,4,1,2,3)

        output=self.model(interpolated_voxels,final_composite,light_position)
        return output*0.5+0.5