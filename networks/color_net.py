import torch
import torch.nn as nn

class NoColor(nn.Module):
    
    def __init__(self):
        super(NoColor, self).__init__()
        
    def forward(self,voxels,text_features):
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
    
    def forward(self,voxels,text_features):
        # voxels are shape [B,N,N,N]
        # output is shape [B,4,N,N,N]
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

class ColorNetv1(nn.Module):
    
    def __init__(self):
        super(ColorNetv1, self).__init__()
        self.text_encoder_1 = nn.Linear(512, 512)
        self.relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.text_encoder_2 = nn.Linear(512, 128)
        

        self.conv1 = nn.Conv3d(129, 128, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(128)
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.conv2 = nn.Conv3d(128, 3, 3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,voxels,text_features):
        # voxels are shape [B,N,N,N]
        # output is shape [B,4,N,N,N]
        text_features = self.text_encoder_1(text_features)
        text_features = self.relu1(text_features)
        text_features = self.text_encoder_2(text_features)
        text_features = text_features.view(-1,128,1,1,1)
        text_features = text_features.expand(-1,-1,128,128,128)

        voxels = voxels.unsqueeze(1)
        full_rep = torch.cat([voxels,text_features],dim=1)

        color_voxels = self.conv1(full_rep)
        # color_voxels = self.bn1(color_voxels)
        color_voxels = self.relu2(color_voxels)
        color_voxels = self.conv2(color_voxels)
        color_voxels = self.sigmoid(color_voxels)
        full_voxels = torch.concatenate([color_voxels,voxels],dim=1)

        return full_voxels

class VoxelNet(nn.Module):
    
    def __init__(self):
        super(VoxelNet, self).__init__()        

        self.conv1 = nn.Conv3d(1, 3, 3, stride=1, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm3d(3)
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.conv2 = nn.Conv3d(3, 3, 3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,voxels,text_features):
        # voxels are shape [B,N,N,N]
        # output is shape [B,4,N,N,N]
        voxels = voxels.unsqueeze(1)
        color_voxels = self.conv1(voxels)
        # color_voxels = self.bn1(color_voxels)
        color_voxels = self.relu2(color_voxels)
        color_voxels = self.conv2(color_voxels)
        color_voxels = self.sigmoid(color_voxels)
        full_voxels = torch.concatenate([color_voxels,voxels],dim=1)

        return full_voxels


if __name__=='__main__':
    out_3d_in = torch.rand(1,128,128,128)
    color_net = ConstantColor()
    out_3d = color_net(out_3d_in)

    out_3d.sum().backward()
    assert color_net.color.grad.norm()>0