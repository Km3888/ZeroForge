from rendering.scripts.renderer import renderer_dict
import rendering.scripts.renderer.transform as dt
import torch

class BaselineRenderer:
    
    def __init__(self,renderer_type,param_dict):
        self.renderer = renderer_dict[renderer_type](param=param_dict)
        self.rotation = dt.Transform(param_dict['device'])
        
    def render(self,volume):
        outputs=[]
        rotated = self.rotation.rotate_random(volume.unsqueeze(1)).squeeze()
        for axis in range(1,4):
            output = self.renderer.render(rotated,axis)
            outputs.append(output)
        
        stacked = torch.cat(outputs,dim=0).double()
        return stacked.unsqueeze(1).expand(-1, 3,-1,-1)    
    
def test_baseline_renderer():
    path="airplane_128.npy"
    import numpy as np
    
    with open(path, 'rb') as f:
        voxel = np.load(f)
    voxel = torch.from_numpy(voxel).float().to('cuda:0')
    voxel.requires_grad=True
    voxel = voxel.unsqueeze(0).unsqueeze(0)
    
    param_dict = {'device':'cuda:0','cube_len':128}
    renderer = BaselineRenderer('absorption_only',param_dict)
    output=renderer.render(voxel)
    return    
    
if __name__=="__main__":
    path="airplane_128.npy"
    import numpy as np
    
    with open(path, 'rb') as f:
        voxel = np.load(f)
    voxel = torch.from_numpy(voxel).float().to('cuda:0')
    voxel.requires_grad=True
    
    renderer = BaselineRenderer()
    output=renderer.render(voxel)