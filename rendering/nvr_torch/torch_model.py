import torch
import torch.nn as nn
import numpy as np
import math

from torch_helpers import load_params,to_txt
from torch_modules import VoxelProcessing,ProjectionProcessing,LightProcessing,\
                            Merger,Decoder,ImageProcessing, ReRendering

class NVR_Plus(nn.Module):
    
    def __init__(self):
        super(NVR_Plus, self).__init__()
        self.voxel_processing = VoxelProcessing()
        self.projection_processing = ProjectionProcessing()
        self.light_processing = LightProcessing()
        self.merger = Merger()
        self.decoder = Decoder()
        self.image_processing = ImageProcessing()
        self.rerenderer = ReRendering()
        
    def forward(self,voxels,final_composite,light_position):
        voxel_representation = self.voxel_processing(voxels)
        projection_representation = self.projection_processing(voxel_representation)
        light_code = self.light_processing(light_position)
        latent_code = self.merger(projection_representation,light_code)
        rendered_image = self.decoder(latent_code)
        composite = self.image_processing(final_composite)
        prediction = self.rerenderer(rendered_image,composite)
        prediction_image = prediction * 0.5 +0.5
        return prediction_image
    
def run_forward():
    model = NVR_Plus()
    params = load_params()
    
    d=torch.save(model.state_dict(),'nvr_plus.pt')
    
    
    # #load .npy files from test_data
    final_composite = np.load('test_data/final_composite.npy')
    final_composite = (final_composite - 0.5)*2
    final_composite = final_composite*2. -1
    interpolated_voxels = np.load('test_data/interpolated_voxels.npy')
    light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773 ]).astype(np.float32)
    light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)
    
    light_position = torch.from_numpy(light_position)
    final_composite = torch.from_numpy(final_composite).permute(0,3,1,2)
    interpolated_voxels = torch.from_numpy(interpolated_voxels).permute(0,4,1,2,3)
    
    model.eval()
    output = model(interpolated_voxels,final_composite,light_position)
    
    to_txt(output)
    permuted_output = output.permute(0,2,3,1) 
    permuted_output = np.clip(permuted_output.detach().numpy(),0,1)
    import matplotlib.pyplot as plt
    plt.imsave('test_data/torch_output.png',permuted_output[0])

    print('break')
    
        

if __name__=="__main__":
    run_forward()