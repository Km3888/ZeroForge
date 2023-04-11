import torch
import torch.nn as nn

from rendering.nvr_torch.torch_model import NVR_Plus
from rendering.preprocess import diff_preprocess,Preprocessor
import numpy as np

class NVR_Renderer(nn.Module):
    
    def __init__(self, args, device):
        super(NVR_Renderer, self).__init__()
        self.model = NVR_Plus()
        self.model.load_state_dict(torch.load(args.nvr_renderer_checkpoint, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
    
    def forward(self,voxels,orthogonal=False,background='default'):
        light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773]).astype(np.float32)
        light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)
        light_position = torch.from_numpy(light_position).to(voxels.device)
        light_position = torch.cat([light_position]*voxels.shape[0],dim=0)
        
        #count how many nan entries are in light_position

        batch_size=voxels.shape[0]
        rotation_angles = self.generate_random_rotations(batch_size,orthogonal).astype(np.float32)

        final_composite,interpolated_voxels = diff_preprocess(voxels,rotation_angles,background=background)
        final_composite = (final_composite - 0.5)*2
        interpolated_voxels = interpolated_voxels.permute(0,4,1,2,3)

        # # self.model = self.model.to(voxels.device)
        output=self.model(interpolated_voxels,final_composite,light_position)
        return output*0.5+0.5

    def generate_random_rotations(self,batch_size,orthogonal):
        if orthogonal:
            assert (batch_size%3==0)
            batch_size= int(batch_size/3)
        # generate a random point on the unit sphere
        u = np.random.uniform(low=0.0,high=1.0,size=(batch_size, 1))
        v = np.random.uniform(low=0.0,high=1.0,size=(batch_size, 1))

        theta = np.arccos(2*u-1)
        phi = 2*np.pi*v
        rotation_z = np.zeros((batch_size,1))

        # convert spherical coordinates to rotation_x, rotation_y, rotation_z
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)

        vectors = np.concatenate([x,y,z],axis=1)
        rotation_angles = np.concatenate([theta,phi,rotation_z],axis=1)
        if not orthogonal:
            return rotation_angles
        
        # generate another set of random rotation angles
        u = np.random.uniform(low=0.0,high=1.0,size=(batch_size, 1))
        v = np.random.uniform(low=0.0,high=1.0,size=(batch_size, 1))

        theta = np.arccos(2*u-1)
        phi = 2*np.pi*v

        # convert spherical coordinates to unit vector
        x_2 = np.sin(theta)*np.cos(phi)
        y_2 = np.sin(theta)*np.sin(phi)
        z_2 = np.cos(theta)

        vectors_2 = np.concatenate([x_2,y_2,z_2],axis=1)
        vectors_2 -= vectors * np.sum(vectors*vectors_2,axis=1,keepdims=True)
        vectors_2 /= np.linalg.norm(vectors_2,axis=1,keepdims=True)

        vectors_3 = np.cross(vectors,vectors_2,axis=1)

        # convert vector_2 and vector_3 to rotation angles
        theta_2 = np.arccos(vectors_2[:,2])
        phi_2 = np.arctan2(vectors_2[:,1],vectors_2[:,0])

        theta_3 = np.arccos(vectors_3[:,2])
        phi_3 = np.arctan2(vectors_3[:,1],vectors_3[:,0])

        rotation_angles_2 = np.concatenate([np.expand_dims(theta_2,1),np.expand_dims(phi_2,1),rotation_z],axis=1)
        rotation_angles_3 = np.concatenate([np.expand_dims(theta_3,1),np.expand_dims(phi_3,1),rotation_z],axis=1)
        
        rotation_angles = np.concatenate([rotation_angles,rotation_angles_2,rotation_angles_3],axis=0)
        rotation_angles *= -1

        return rotation_angles