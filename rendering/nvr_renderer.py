import torch

from rendering.nvr_torch.torch_model import NVR_Plus
from rendering.preprocess import diff_preprocess
import numpy as np

class NVR_Renderer:
    
    def __init__(self):
        self.model = NVR_Plus()
        self.model.load_state_dict(torch.load('rendering/nvr_torch/nvr_plus.pt'))
        self.model.to('cuda:0') #TODO make device arbitrary
        self.model.eval()
    
    def render(self,voxels,angle=139.):
        light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773]).astype(np.float32)
        light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)
        light_position = torch.from_numpy(light_position).to(voxels.device)
        light_position = torch.cat([light_position]*voxels.shape[0],dim=0)

        batch_size=voxels.shape[0]
        angle=np.random.uniform(low=0.0,high=180.0,size=(batch_size,1)).astype(np.float32)
        final_composite,interpolated_voxels = diff_preprocess(voxels,angle)
        final_composite = (final_composite - 0.5)*2
        interpolated_voxels = interpolated_voxels.permute(0,4,1,2,3)

        output=self.model(interpolated_voxels,final_composite,light_position)
        return output*0.5+0.5
