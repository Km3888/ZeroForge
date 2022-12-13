import torch

class NVR(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, voxel):
        ctx.save_for_backward(voxel)
        return None
    
    @staticmethod
    def backward(ctx, grad_output):
        voxel, = ctx.saved_tensors
        return grad_output
    