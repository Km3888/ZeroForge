import torch
import torch.nn as nn

class NoColor(nn.Module):
    
    def __init__(self):
        super(NoColor, self).__init__()
        
    def forward(self,voxels):
        voxels = voxels.unsqueeze(1)
        colors = voxels.repeat(1,3,1,1,1)/3
        voxels = torch.concatenate([colors,voxels],dim=1)
        return voxels

class ConstantColor(nn.Module):
    
    def __init__(self,color=None):
        super(ConstantColor, self).__init__()
        if color is None:
            color = 0.5 * torch.ones(3)
        self.color = nn.Parameter(color)
    
    def forward(self,voxels):
        # voxels are shape [B,N,N,N]
        # output is shape [B,3,N,N,N]
        batch_size = voxels.shape[0]
        # add color channel
        voxels = voxels.unsqueeze(1)
        colors = self.color
        #making it the same color for every voxel
        colored_voxels = colors.view(1,3,1,1,1).expand(batch_size,3,128,128,128)
        
        out_voxels = torch.concatenate([colored_voxels,voxels],dim=1) 
        return out_voxels
    
class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self,voxels):
        return voxels

if __name__=='__main__':
    out_3d_in = torch.rand(1,128,128,128)
    color_net = ConstantColor()
    out_3d = color_net(out_3d_in)

    out_3d.sum().backward()
    assert color_net.color.grad.norm()>0