import torch

from rendering.nvr_torch.torch_model import NVR
from rendering.preprocess import diff_preprocess
import numpy as np

class NVR_Renderer:
    
    def __init__(self):
        self.model = NVR()
        self.model.load_state_dict(torch.load('rendering/nvr_torch/nvr_plus.pth'))
        
    
    def render(self,voxels,angle=139.):
        batch_size=voxels.shape[0]
        angle=np.random.uniform(low=0.0,high=180.0,size=(batch_size,1)).astype(np.float32)
        final_composite,interpolated_voxels = diff_preprocess(voxels,angle)
        output=self.model(final_composite,interpolated_voxels)
        return output
