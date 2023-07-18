import torch
import torch.nn as nn



class NoColor(nn.Module):
    
    def __init__(self):
        super(NoColor, self).__init__()
        
    def forward(self,voxels):
        voxles = voxels.unsqueeze(1)
        voxels = voxels.repeat(1,3,1,1,1)/3
        return voxels

class ConstantColor(nn.Module):
    
    def __init__(self,color=None):
        super(ConstantColor, self).__init__()
        if color is None:
            color = torch.ones(3)
        self.color = nn.Parameter(color)
    
    def forward(self,voxels):
        # voxels are shape [B,N,N,N]
        # output is shape [B,3,N,N,N]
        
        batch_size = voxels.shape[0]
        # add color channel
        voxels = voxels.unsqueeze(1)
        voxels = voxels.repeat(1,3,1,1,1)
        
        # multiply by color value
        normed_color = self.color / self.color.norm()
        voxels = voxels * normed_color.view(1,3,1,1,1)
        return voxels