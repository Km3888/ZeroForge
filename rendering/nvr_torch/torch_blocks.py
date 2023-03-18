import torch
import torch.nn as nn

class ResBlock2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        
        self.convs = [self.conv1,self.conv2]
        self.bns = [self.bn1,self.bn2]
        if stride!= 1:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, 1, 0) 
            # TODO may need padding
            self.bn3 = nn.BatchNorm2d(out_channels)
            
            self.convs.append(self.conv3)
            self.bns.append(self.bn3)
    
    def forward(self,x,extra=False):
        x = x.to(self.conv1.weight.device)
        shortcut = x
        assert x.device == self.conv1.weight.device
        # print(x.device,shortcut.device,self.conv1.weight.device,self.bn1.weight.device)
        x = self.conv1(x)
        if len(self.convs)>2:
            x = x[:,:,1::2,1::2]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if len(self.convs)>2:
            shortcut = self.conv3(shortcut)
            shortcut = shortcut[:,:,::2,::2]
            x = self.bn3(x)
            
        x = x + shortcut
        x = self.relu(x)
        debug_output = x
        
        if extra:
            return x,debug_output
        
        return x
    
class ResBlock3d(nn.Module):
    
    def __init__(self,input_dim,nfilters):
        super(ResBlock3d,self).__init__()
        self.conv_1 = nn.Conv3d(input_dim,nfilters,3,stride=1,padding=1,bias=True)
        self.bn1 = nn.BatchNorm3d(nfilters)
        
        self.conv_2 = nn.Conv3d(nfilters,nfilters,3,stride=1,padding=1,bias=True)
        self.bn2 = nn.BatchNorm3d(nfilters)
        
        self.relu = nn.LeakyReLU(negative_slope=0.3)

        self.convs = [self.conv_1,self.conv_2]
        self.bns = [self.bn1,self.bn2]
    
    def forward(self,x):
        shortcut = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv_2(x) # 0.9887371
        x = self.bn2(x) # 0.9894749
        x = x + shortcut # 0.99114084
        
        x = self.relu(x)
        return x
        
class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, nfilters, size, strides,
                  alpha_lrelu=0.2, normalization='None', relu=True):
        super(ConvBlock3d, self).__init__()

        # set padding so that output shape is the same as input shape
        padding = (size-1)//2
        self.stride = strides
        
        self.conv = nn.Conv3d(in_channels, nfilters, size, strides, padding=padding, padding_mode="zeros",bias=False)
        self.bn = nn.BatchNorm3d(nfilters)
        self.relu = nn.LeakyReLU(negative_slope=alpha_lrelu)

    def forward(self, x):
        if self.stride == 1:
            x=nn.functional.pad(x,(0,1,0,1,0,1),"constant",0)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ConvBlock2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(ConvBlock2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,1,padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self,x):
        x = nn.functional.pad(x,(0,1,0,1),"constant",0)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvTransposeBlock2d(nn.Module):
    
    def __init__(self,in_channels,out_channels,strides):
        super(ConvTransposeBlock2d, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels,out_channels,4,stride=2,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self,z):
        z = self.conv2d(z)[:,:,1:-1,1:-1]
        z = self.bn(z)
        z = self.relu(z)
        return z
    
class UpConv(nn.Module):
    
    def __init__(self,in_features,out_features,kernel):
        super(UpConv, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv_layer = nn.Conv2d(in_features,out_features,kernel,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self,x):
        x = self.upsampler(x)
        x = nn.functional.pad(x,(0,1,0,1))
        x = self.conv_layer(x)
        x = self.relu(x)
        
        return x

class UNet(nn.Module):
    
    def __init__(self,in_features,out_features):
        super(UNet, self).__init__()
        
        self.res_1 = ResBlock2d(in_features,128,stride=2)
        self.res_2 = ResBlock2d(128,256,stride=2)
        self.res_3 = ResBlock2d(256,512,stride=2)
        
        self.res_4 = ResBlock2d(512,512,stride=1)
        self.res_5 = ResBlock2d(512,512,stride=1)
        self.res_6 = ResBlock2d(512,512,stride=1)
        
        self.upconv_1 = UpConv(512,256,4)
        self.upconv_2 = UpConv(256,128,4)
        self.upconv_3 = UpConv(128,64,4)
        
        self.conv_1 = nn.Conv2d(512,256,4,1,padding=1)
        self.conv_2 = nn.Conv2d(256,128,4,1,padding=1)
        self.conv_3 = nn.Conv2d(96,out_features,4,1,padding=1)
        
        self.res_blocks = [self.res_1,self.res_2,self.res_3,self.res_4,self.res_5,self.res_6]
        self.upconv_blocks = [self.upconv_1,self.upconv_2,self.upconv_3]
        self.conv_layers = [self.conv_1,self.conv_2,self.conv_3]
        
        
    def forward(self,x,extra=False):
        
        e1 = self.res_1(x)
        e2 = self.res_2(e1)
        e3 = self.res_3(e2)
        
        mid1 = self.res_4(e3)
        mid2 = self.res_5(mid1)
        mid3 = self.res_6(mid2)
        
        d0 = self.upconv_1(mid3)
        d1 = torch.cat([d0,e2],dim=1)
        d1 = nn.functional.pad(d1,(0,1,0,1))
        d2 = self.conv_1(d1)
        
        d3 = self.upconv_2(d2)
        d4 = torch.cat([d3,e1],dim=1)
        d4 = nn.functional.pad(d4,(0,1,0,1))
        d5 = self.conv_2(d4)
        
        d6 = self.upconv_3(d5)
        d7 = torch.cat([d6,x],dim=1)
        d7 = nn.functional.pad(d7,(0,1,0,1))
        d8 = self.conv_3(d7)
        
        return d8
    