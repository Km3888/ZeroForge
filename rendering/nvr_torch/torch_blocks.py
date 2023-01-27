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
        shortcut = x
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
    
    def parameterize(self,param_dict,module,conv_id,bn_id):
        
        for i,conv_i in enumerate(self.convs):
            conv_weight = param_dict[module+'/conv2d_%s/kernel' % (conv_id+i+1)]
            conv_bias = param_dict[module+'/conv2d_%s/bias' % (conv_id+i+1)]
            
            conv_weight = torch.from_numpy(conv_weight).permute(3,2,0,1)
            conv_bias = torch.from_numpy(conv_bias)
            
            assert conv_weight.shape == conv_i.weight.shape
            assert conv_bias.shape == conv_i.bias.shape
            
            conv_i.weight.data = conv_weight
            conv_i.bias.data = conv_bias
            
        for i,bn_i in enumerate(self.bns):
            bn_gamma = param_dict[module+'/batch_normalization_%s/gamma' % (bn_id+i)]
            bn_beta = param_dict[module+'/batch_normalization_%s/beta' % (bn_id+i)]
            bn_mean = param_dict[module+'/batch_normalization_%s/moving_mean' % (bn_id+i)]
            bn_var = param_dict[module+'/batch_normalization_%s/moving_variance' % (bn_id+i)]
            
            bn_gamma = torch.from_numpy(bn_gamma)
            bn_beta = torch.from_numpy(bn_beta)
            bn_mean = torch.from_numpy(bn_mean)
            bn_var = torch.from_numpy(bn_var)
            
            assert bn_i.weight.shape == bn_gamma.shape
            assert bn_i.bias.shape == bn_beta.shape
            assert bn_i.running_mean.shape == bn_mean.shape
            assert bn_i.running_var.shape == bn_var.shape
            
            bn_i.weight.data = bn_gamma
            bn_i.bias.data = bn_beta
            bn_i.running_mean.data = bn_mean
            bn_i.running_var.data = bn_var

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
        
    
    def parameterize(self,params_dict,layer_id):
        module = 'Network/VoxelProcessing'
        for i,conv in enumerate(self.convs):
            conv_weight = params_dict[module+'/conv3d_%s/kernel' % (layer_id+i)]
            conv_weight = torch.from_numpy(conv_weight).permute(4,3,0,1,2)
            conv_weight = torch.nn.parameter.Parameter(conv_weight)
            
            assert conv_weight.shape == conv.weight.shape
            conv.weight = conv_weight

            conv_bias = params_dict[module+'/conv3d_%s/bias' % (layer_id+i)]
            conv_bias = torch.from_numpy(conv_bias)
            conv_bias = torch.nn.parameter.Parameter(conv_bias)
            
            assert conv_bias.shape == conv.bias.shape
            conv.bias = conv_bias
            
        for i,bn in enumerate(self.bns):
            bn.running_mean = torch.from_numpy(params_dict[module+'/batch_normalization_%s/moving_mean' % (layer_id+i)])
            bn.running_var = torch.from_numpy(params_dict[module+'/batch_normalization_%s/moving_variance' % (layer_id+i)])
            
            tf_gamma = torch.from_numpy(params_dict['%s/batch_normalization_%s/gamma' % (module,layer_id+i)])
            tf_beta = torch.from_numpy(params_dict['%s/batch_normalization_%s/beta' % (module,layer_id+i)])
            
            tf_gamma = torch.nn.parameter.Parameter(tf_gamma)
            tf_beta = torch.nn.parameter.Parameter(tf_beta)
            
            bn.weight = tf_gamma
            bn.bias = tf_beta

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
    
    def parameterize(self,params,layer_id):
        # setting convolution weights
        module = 'Network/VoxelProcessing'
        conv_weight = params[module+'/conv3d%s/kernel' % layer_id]
        conv_weight = torch.from_numpy(conv_weight).permute(4,3,0,1,2)
        conv_weight = torch.nn.parameter.Parameter(conv_weight)
        
        assert conv_weight.shape == self.conv.weight.shape
        self.conv.weight = conv_weight
        
        # setting batch norm params
        self.bn.running_mean = torch.from_numpy(params[module+'/batch_normalization%s/moving_mean' % layer_id])
        self.bn.running_var = torch.from_numpy(params[module+'/batch_normalization%s/moving_variance' % layer_id])
        
        tf_gamma = torch.from_numpy(params['%s/batch_normalization%s/gamma' % (module,layer_id)])
        tf_beta = torch.from_numpy(params['%s/batch_normalization%s/beta' % (module,layer_id)])
        
        tf_gamma = torch.nn.parameter.Parameter(tf_gamma)
        tf_beta = torch.nn.parameter.Parameter(tf_beta)
        
        self.bn.weight = tf_gamma
        self.bn.bias = tf_beta

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

    def parameterize(self,params,module,conv_id,bn_id):
        self.conv.weight.data = torch.from_numpy(params[module + '/conv2d_%s/kernel' % conv_id]).permute(3,2,0,1)
        self.bn.weight.data = torch.from_numpy(params[module + '/batch_normalization_%s/gamma' % bn_id])
        self.bn.bias.data = torch.from_numpy(params[module + '/batch_normalization_%s/beta' % bn_id])
        self.bn.running_mean.data = torch.from_numpy(params[module + '/batch_normalization_%s/moving_mean' % bn_id])
        self.bn.running_var.data = torch.from_numpy(params[module + '/batch_normalization_%s/moving_variance' % bn_id])

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
    
    def parameterize(self,params_dict,layer_id,bn_id):
        module = 'Network/Decoder'
        
        conv_weight = nn.Parameter(torch.from_numpy(params_dict[module+'/conv2d_transpose%s/kernel' % layer_id]).permute(3,2,0,1))
        assert self.conv2d.weight.shape == conv_weight.shape
        self.conv2d.weight = conv_weight
        
        #parameterize batch norm
        bn_gamma = nn.Parameter(torch.from_numpy(params_dict[module+'/batch_normalization_%s/gamma' % bn_id]))
        bn_beta = nn.Parameter(torch.from_numpy(params_dict[module+'/batch_normalization_%s/beta' % bn_id]))
        bn_mean = torch.from_numpy(params_dict[module+'/batch_normalization_%s/moving_mean' % bn_id])
        bn_var = torch.from_numpy(params_dict[module+'/batch_normalization_%s/moving_variance' % bn_id])
        
        assert self.bn.weight.shape == bn_gamma.shape
        self.bn.weight = bn_gamma
        assert self.bn.bias.shape == bn_beta.shape
        self.bn.bias = bn_beta
        assert self.bn.running_mean.shape == bn_mean.shape
        self.bn.running_mean = bn_mean
        assert self.bn.running_var.shape == bn_var.shape
        self.bn.running_var = bn_var

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
    
    def parameterize(self,params_dict,module,conv_id,bn_id):
        
        conv_weight = torch.from_numpy(params_dict[module+'/conv2d_'+str(conv_id)+'/kernel']).permute(3,2,0,1)
        
        assert self.conv_layer.weight.data.shape == conv_weight.shape
        
        self.conv_layer.weight.data = conv_weight

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
    
    def parameterize(self,params_dict,conv_id,bn_id):
        
        for i,block in enumerate(self.res_blocks):
            block.parameterize(params_dict,'Network/NeuralRerenderingNetwork',conv_id,bn_id)
            
            conv_id += len(block.convs)
            bn_id += len(block.bns)
        
        conv_id += 1
        
        for upconv,conv in zip(self.upconv_blocks,self.conv_layers):
            upconv.parameterize(params_dict,'Network/NeuralRerenderingNetwork',conv_id,bn_id)
            conv_id+=1
            conv.weight.data = torch.from_numpy(params_dict['Network/NeuralRerenderingNetwork/conv2d_'+str(conv_id)+'/kernel']).permute(3,2,0,1)
            conv.bias.data = torch.from_numpy(params_dict['Network/NeuralRerenderingNetwork/conv2d_'+str(conv_id)+'/bias'])
            conv_id+=1
