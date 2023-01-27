import torch
import torch.nn as nn

from torch_helpers import load_params
from torch_blocks import ConvBlock3d, ConvBlock2d, ResBlock3d, ResBlock2d, \
                        ConvTransposeBlock2d, UNet

class VoxelProcessing(nn.Module):
    def __init__(self):
        super(VoxelProcessing,self).__init__()
        nf_2d = 512
        self.vol0_a = ConvBlock3d(4, 16, size=4, strides=2)
        self.vol0_b = ConvBlock3d(16, 16, size=4, strides=1)
        self.vol1_a = ConvBlock3d(16, 16, size=4, strides=2)
        self.vol1_b = ConvBlock3d(16, 32, size=4, strides=1)
        self.vol1_c = ConvBlock3d(32, 32, size=4, strides=1)
        self.conv3ds = [self.vol0_a, self.vol0_b, self.vol1_a, self.vol1_b, self.vol1_c]
        
        self.vol_a1 = ResBlock3d(32, 32)
        self.vol_a2 = ResBlock3d(32, 32)
        self.vol_a3 = ResBlock3d(32, 32)
        self.vol_a4 = ResBlock3d(32, 32)
        self.vol_a5 = ResBlock3d(32, 32)
        
        self.resblocks = [self.vol_a1, self.vol_a2, self.vol_a3, self.vol_a4, self.vol_a5]
        
        self.vol_encoder = nn.Conv2d(32, nf_2d, 1, 1, 0)
        self.final_relu = nn.LeakyReLU(negative_slope=0.3)
        
        self.parameterize()
        
    def forward(self, voxels):
        voxels = self.vol0_a(voxels)
        voxels = self.vol0_b(voxels)
        voxels = self.vol1_a(voxels) 
        voxels = self.vol1_b(voxels) 
        voxels = self.vol1_c(voxels) # 0.9999977
        
        shortcut = voxels
        
        voxels = self.vol_a1(voxels) # 0.9905581
        voxels = self.vol_a2(voxels) # 0.9871442
        voxels = self.vol_a3(voxels)
        voxels = self.vol_a4(voxels)
        voxels = self.vol_a5(voxels)
        
        voxels = voxels + shortcut
        voxels = voxels.permute(0,2,3,4,1)
        voxels = voxels.reshape(voxels.shape[0],32,32,-1)
        voxels = voxels.permute(0,3,1,2)
        voxels = self.vol_encoder(voxels)
        voxels = self.final_relu(voxels)
        return voxels
    
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        i=0
        for conv in self.conv3ds:
            if i==0:
                conv.parameterize(params_dict,'')
            else:
                conv.parameterize(params_dict,'_%s' % i)
            i+=1
            
        for res in self.resblocks:
            res.parameterize(params_dict,i)
            i+=2
            
        self.vol_encoder.weight = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/VoxelProcessing/conv2d/kernel']).permute(3,2,0,1))
        self.vol_encoder.bias = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/VoxelProcessing/conv2d/bias']))

class ProjectionProcessing(nn.Module):
    
    def __init__(self):
        super(ProjectionProcessing, self).__init__()
        self.e1 = ResBlock2d(512,512)
        self.e2 = ResBlock2d(512,512)
        self.e3 = ResBlock2d(512,512)
        self.e4 = ResBlock2d(512,512)
        self.e5 = ResBlock2d(512,512)
        
        self.e_blocks = [self.e1,self.e2,self.e3,self.e4,self.e5]
        self.parameterize()
        
    def forward(self,x):
        shortcut = x
        
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        
        x = x + shortcut
        return x
        
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        
        for i,e_i in enumerate(self.e_blocks):
            e_i.parameterize(params_dict,'Network/ProjectionProcessing',2*i,2*i+15)

class LightProcessing(nn.Module):
    
    def __init__(self):
        super(LightProcessing, self).__init__()
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,64)

        self.parameterize()
        
    def forward(self,light):
        fc_light = self.fc1(light)
        light_code = self.fc2(fc_light)
        light_code = light_code.repeat((1,32*32))
        light_code = light_code.reshape((-1,32,32,64))
        light_code = light_code.permute((0,3,1,2))
        return light_code
    
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        
        self.fc1.weight = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/LightProcessing/dense/kernel']).permute(1,0))
        self.fc1.bias = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/LightProcessing/dense/bias']))
        
        self.fc2.weight = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/LightProcessing/dense_1/kernel']).permute(1,0))
        self.fc2.bias = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/LightProcessing/dense_1/bias']))

class Merger(nn.Module):

    def __init__(self):
        super(Merger,self).__init__()
        self.conv1 = ConvBlock2d(512+64,512,4,1)

        self.res1 = ResBlock2d(512,512)
        self.res2 = ResBlock2d(512,512)
        self.res3 = ResBlock2d(512,512)
        self.res4 = ResBlock2d(512,512)
        self.res5 = ResBlock2d(512,512)

        self.res_blocks = [self.res1,self.res2,self.res3,self.res4,self.res5]

        self.parameterize()

    def forward(self,proj_representation,light_code):
        latent_code = torch.cat((proj_representation,light_code),dim=1)
        latent_code = self.conv1(latent_code)

        shortcut = latent_code
        
        for res_i in self.res_blocks:
            latent_code = res_i(latent_code)
        
        latent_code = latent_code + shortcut

        return latent_code

    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        module = 'Network/Merger'

        self.conv1.parameterize(params_dict,module,11,25)
        for i,res_i in enumerate(self.res_blocks):
            res_i.parameterize(params_dict,module,11+2*i,26+2*i)

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_t1 = ConvTransposeBlock2d(512,128,2)
        self.conv_t2 = ConvTransposeBlock2d(128,64,2)
        self.conv_t3 = ConvTransposeBlock2d(64,32,2)
        
        self.conv_1 = ConvBlock2d(128,128,4,1)
        self.conv_2 = ConvBlock2d(64,64,4,1)
        self.conv_3 = ConvBlock2d(32,32,4,1)
        
        self.final_conv = nn.Conv2d(32,32,4,padding=1,bias=False)
        
        self.tranpose_blocks = [self.conv_t1,self.conv_t2,self.conv_t3]
        self.conv_blocks = [self.conv_1,self.conv_2,self.conv_3]
        
        self.parameterize()
        
    def forward(self,z):
        z = self.conv_t1(z)
        z = self.conv_1(z)
        
        z = self.conv_t2(z)
        z = self.conv_2(z)
        
        z = self.conv_t3(z)
        z = self.conv_3(z)
        
        z = nn.functional.pad(z,(0,1,0,1))
        z = self.final_conv(z)
        
        return z
    
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        
        for i,block_i in enumerate(self.tranpose_blocks):
            if i==0:
                block_i.parameterize(params_dict,'',36)
            else:
                block_i.parameterize(params_dict,'_%s'%i,36+i*2)
                
        for i, block_i in enumerate(self.conv_blocks):
            block_i.parameterize(params_dict,'Network/Decoder',22+i,37+i*2)
            
        conv_weight = nn.Parameter(torch.from_numpy(params_dict['Network/Decoder/conv2d_25/kernel']).permute(3,2,0,1))
        assert conv_weight.shape == self.final_conv.weight.shape
        self.final_conv.weight = conv_weight

class ImageProcessing(nn.Module):
    
    def __init__(self):
        super(ImageProcessing, self).__init__()
        
        self.conv_1 = ConvBlock2d(3,32,4,1)
        self.conv_2 = ConvBlock2d(32,32,4,1)
        
        self.conv_blocks = [self.conv_1,self.conv_2]
        
        self.parameterize()
        
    def forward(self,x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        
        for i,conv in enumerate(self.conv_blocks):
            conv.parameterize(params_dict,'Network/ImageProcessingNetwork',26+i,42+i)

class ReRendering(nn.Module):
    
    def __init__(self):
        super(ReRendering, self).__init__()
        self.u_net = UNet(32,32)
        self.conv_block = ConvBlock2d(32,32,4,1)
        self.conv_layer = nn.Conv2d(32,3,4,1,padding=1)
        
        self.parameterize()
    
    def forward(self,rendering,composite):
        x = rendering + composite
        x = self.u_net(x,extra=False)
        x = self.conv_block(x)
        x = nn.functional.pad(x,(0,1,0,1))
        x = self.conv_layer(x)
        return x
    
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        
        self.u_net.parameterize(params_dict,27,44)
        self.conv_block.parameterize(params_dict,'Network/NeuralRerenderingNetwork',49,59)
        
        self.conv_layer.weight = nn.Parameter(torch.from_numpy(params_dict['Network/NeuralRerenderingNetwork/conv2d_50/kernel']).permute(3,2,0,1))
