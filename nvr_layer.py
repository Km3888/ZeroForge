import torch
import numpy as np

import tensorflow as tf

from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer import models
from tensorflow_graphics.rendering.volumetric import visual_hull

camera_rotation_matrix= [[ 9.9997330e-01,  7.3080887e-03,  8.9461202e-11],\
    [ 4.9256836e-03, -6.7398632e-01, -7.3872751e-01],\
    [-5.3986851e-03,  7.3870778e-01, -6.7400432e-01]]
camera_translation_vector = [[5.2963998e-09],\
    [5.3759331e-01],\
    [4.2457557e+00]]
focal = [284.44446, 284.44446]
principal_point = [128., 128.]
light_position = [-1.0901234 ,  0.01720496,  2.6110773 ]
object_rotation = [139.]
object_translation = [-0.39401,  0.,  0.4]
object_elevation = [47.62312]

class NVR(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, voxel):
        ctx.save_for_backward(voxel)
        return torch.ones((128,128))
    
    @staticmethod
    def backward(ctx, grad_output):
        voxel, = ctx.saved_tensors
        return torch.ones((128,128,128))


    
if __name__=="__main__":
    #load airplane voxel using numpy
    path="airplane_128.npy"
    with open(path, 'rb') as f:
        voxel = np.load(f)
    voxel = torch.from_numpy(voxel).float()
    voxel.requires_grad=True
    
    output = NVR.apply(voxel)
    print(output.shape)
    loss = output.sum()
    loss.backward()
    print(voxel.grad.shape)