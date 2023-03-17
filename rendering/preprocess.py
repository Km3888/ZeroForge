import argparse
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer import models
from tensorflow_graphics.rendering.volumetric import visual_hull

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d

if True:
    camera_rotation_matrix= np.array([[ 9.9997330e-01,  7.3080887e-03,  8.9461202e-11],\
    [ 4.9256836e-03, -6.7398632e-01, -7.3872751e-01],\
    [-5.3986851e-03,  7.3870778e-01, -6.7400432e-01]]).astype(np.float32)
    camera_translation_vector = np.array([[5.2963998e-09],\
        [5.3759331e-01],\
        [4.2457557e+00]]).astype(np.float32)
    focal = np.array([284.44446, 284.44446]).astype(np.float32)
    principal_point = np.array([128., 128.]).astype(np.float32)
    light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773 ]).astype(np.float32)
    object_translation = np.array([-0.39401,  0.,  0.4]).astype(np.float32)
    object_elevation = np.array([47.62312]).astype(np.float32)
    
    camera_rotation_matrix=np.expand_dims(camera_rotation_matrix,axis=(0))
    camera_translation_vector=np.expand_dims(camera_translation_vector,axis=(0)).astype(np.float32)
    light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)

    object_translation=np.expand_dims(object_translation,axis=(0,1))
    object_elevation=np.expand_dims(object_elevation,axis=(0))

    #convert object arrays to float32 datatype
    object_translation=object_translation.astype(np.float32)
    object_elevation=object_elevation.astype(np.float32)


    VOXEL_SIZE = 128
    IMAGE_SIZE = 256
    GROUND_COLOR = np.array((136., 162, 199))/255.
    BLENDER_SCALE = 2
    DIAMETER = 4.2  # The voxel area in world coordinates
    
    # device = 'cuda:0'

def process_voxel(voxel):
    voxel = voxel.detach().cpu().numpy()
    edited_voxel = np.transpose(voxel,(1,0,2)) 
    edited_voxel = np.flip(edited_voxel,0)
    
    edited_voxel = edited_voxel.reshape((1,128,128,128,1))
    expanded = np.repeat(edited_voxel,4,axis=4)/3
    expanded[:,:,:,:,-1]=3*expanded[:,:,:,:,0]
    
    return expanded

def make_arrays(angle):
    #helper function for estimate_ground_image
    object_rotation_v = angle
    object_translation_v = object_translation[:, 0, [1, 0, 2]]*BLENDER_SCALE
    object_elevation_v = object_elevation

    ground_occupancy = np.zeros((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1),
                                dtype=np.float32)
    ground_occupancy[-2, 1:-2, 1:-2, 0] = 1

    ground_voxel_color = np.ones((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 3), 
                                dtype=np.float32)*\
                        np.array(GROUND_COLOR, dtype=np.float32)
    ground_voxel_color = np.concatenate([ground_voxel_color, ground_occupancy],
                                        axis=-1)

    euler_angles_x = np.deg2rad(180-object_rotation_v)*np.array([1, 0, 0],
                                                                dtype=np.float32)
    euler_angles_y = np.deg2rad(90-object_elevation_v)*np.array([0, 1, 0],
                                                                dtype=np.float32)
    translation_vector = (object_translation_v/(DIAMETER*0.5))

    return ground_occupancy,ground_voxel_color,euler_angles_x,euler_angles_y,translation_vector
    
def estimate_ground_image(object_voxels,angle):
    ground_occupancy,ground_voxel_color,euler_angles_x,\
        euler_angles_y,translation_vector = make_arrays(angle)
    
    scene_voxels = object_voxels*(1-ground_occupancy) + \
                    ground_voxel_color*ground_occupancy

    interpolated_voxels = helpers.object_to_world(scene_voxels,
                                                euler_angles_x,
                                                euler_angles_y,
                                                translation_vector)
    return interpolated_voxels

def og_preprocess(object_voxels,angle=139):
    object_voxels = process_voxel(object_voxels)
    
    interpolated_voxels = estimate_ground_image(object_voxels,angle)

    ground_image, ground_alpha = \
        helpers.generate_ground_image(IMAGE_SIZE, IMAGE_SIZE, focal, principal_point,
                            camera_rotation_matrix,
                            camera_translation_vector[:, :, 0],
                            GROUND_COLOR)
    object_rotation_dvr = np.array(np.deg2rad(angle),
                            dtype=np.float32)
    object_translation_dvr = np.array(object_translation[..., [0, 2, 1]], 
                                    dtype=np.float32)
    object_translation_dvr -= np.array([0, 0, helpers.OBJECT_BOTTOM],
                                        dtype=np.float32)

    rerendering = \
    helpers.render_voxels_from_blender_camera(object_voxels,
                                        object_rotation_dvr,
                                        object_translation_dvr,
                                        256, 
                                        256,
                                        focal,
                                        principal_point,
                                        camera_rotation_matrix,
                                        camera_translation_vector,
                                        absorption_factor=1.0,
                                        cell_size=1.1,
                                        depth_min=3.0,
                                        depth_max=5.0,
                                         frustum_size=(128, 128, 128))
    rerendering_image, rerendering_alpha = tf.split(rerendering, [3, 1], axis=-1)

    rerendering_image = tf.image.resize(rerendering_image, (256, 256))
    rerendering_alpha = tf.image.resize(rerendering_alpha, (256, 256))

    BACKGROUND_COLOR = 0.784
    final_composite = BACKGROUND_COLOR*(1-rerendering_alpha)*(1-ground_alpha) + \
                    ground_image*(1-rerendering_alpha)*ground_alpha + \
                    rerendering_image*rerendering_alpha
    
    return final_composite,interpolated_voxels

def diff_object_to_world(voxels,
                    euler_angles_x,
                    euler_angles_y,
                    translation_vector,
                    target_volume_size=(128, 128, 128)):
    """Apply the transformations to the voxels and place them in world coords."""
    scale_factor = 1.82  # object to world voxel space scale factor

    translation_vector = tf.expand_dims(translation_vector, axis=-1)

    sampling_points = tf.cast(helpers.sampling_points_from_3d_grid(target_volume_size),
                            tf.float32)  # 128^3 X 3
    transf_matrix_x = rotation_matrix_3d.from_euler(euler_angles_x)  # [B, 3, 3]
    transf_matrix_y = rotation_matrix_3d.from_euler(euler_angles_y)  # [B, 3, 3]
    transf_matrix = tf.matmul(transf_matrix_x, transf_matrix_y)  # [B, 3, 3]
    transf_matrix = transf_matrix*scale_factor  # [B, 3, 3]
    sampling_points = tf.matmul(transf_matrix,
                                tf.transpose(sampling_points))  # [B, 3, N]
    translation_vector = tf.matmul(transf_matrix, translation_vector)  # [B, 3, 1]
    sampling_points = sampling_points - translation_vector
    sampling_points = tf.linalg.matrix_transpose(sampling_points)
    sampling_points = tf.cast(sampling_points, tf.float32)

    #okay now we care about differentiability so convert to PyTorch and put on GPU
    sampling_points = torch.from_numpy(sampling_points.numpy()).to(voxels.device)

    sampling_points = sampling_points.view(-1,128,128,128,3)
    sampling_points = sampling_points.expand(voxels.shape[0],-1,-1,-1,-1)

    voxels = voxels.permute(0,4,1,2,3)

    interpolated_points = torch.nn.functional.grid_sample(voxels, sampling_points,mode='bilinear')
    
    return interpolated_points.permute(0,2,3,4,1)    # return interpolated_voxels

def diff_estimate_ground_image(object_voxels,angle):    
    ground_occupancy,ground_voxel_color,euler_angles_x,\
        euler_angles_y,translation_vector = make_arrays(angle)
        
    #convert arrays to pytorch and put them on cuda:0
    ground_occupancy = torch.from_numpy(ground_occupancy).to(object_voxels.device)
    ground_voxel_color = torch.from_numpy(ground_voxel_color).to(object_voxels.device)
    scene_voxels = object_voxels*(1-ground_occupancy) + \
                    ground_voxel_color*ground_occupancy
    
    interpolated_voxels = diff_object_to_world(scene_voxels,
                                                euler_angles_x,
                                                euler_angles_y,
                                                translation_vector)
    return interpolated_voxels
    
def diff_load_voxel(voxel):
    edited_voxel = torch.transpose(voxel,2,1)
    edited_voxel = torch.flip(edited_voxel,[0])
    edited_voxel = edited_voxel.view(-1,128,128,128,1)
    
    
    expanded = edited_voxel.repeat(1,1,1,1,4)/3
    expanded[:,:,:,:,-1]=3*expanded[:,:,:,:,0]
    
    return expanded

def diff_render_voxels_from_blender_camera(voxels,
                                        object_rotation,
                                        object_translation,
                                        height,
                                        width,
                                        focal,
                                        principal_point,
                                        camera_rotation_matrix,
                                        camera_translation_vector,
                                        frustum_size=(256, 256, 512),
                                        absorption_factor=0.1,
                                        cell_size=1.0,
                                        depth_min=0.0,
                                        depth_max=5.0):
    """Renders the voxels according to their position in the world."""
    batch_size = voxels.shape[0]
    voxel_size = voxels.shape[1]
    sampling_volume = helpers.sampling_points_from_frustum(height,
                                                    width,
                                                    focal,
                                                    principal_point,
                                                    depth_min=depth_min,
                                                    depth_max=depth_max,
                                                    frustum_size=frustum_size)
    sampling_volume = \
    helpers.place_frustum_sampling_points_at_blender_camera(sampling_volume,
                                                    camera_rotation_matrix,
                                                    camera_translation_vector)
    interpolated_voxels = \
    diff_object_rotation_in_blender_world(voxels, object_rotation)

    # Adjust the camera (translate the camera instead of the object)
    sampling_volume = sampling_volume - object_translation
    sampling_volume = sampling_volume/helpers.CUBE_BOX_DIM
    sampling_tensor = torch.from_numpy(sampling_volume.numpy()).to(voxels.device)
    sampling_tensor = sampling_tensor.view(-1,128,128,128,3)
    
    interpolated_voxels = interpolated_voxels.permute(0,4,1,2,3)

    sampling_tensor = sampling_tensor.expand(batch_size,-1,-1,-1,-1)
    camera_voxels = torch.nn.functional.grid_sample(interpolated_voxels, sampling_tensor,mode='bilinear')
    camera_voxels = camera_voxels.permute(0,2,3,4,1)
    voxel_image = ea_render(camera_voxels,
                                            absorption_factor=absorption_factor,
                                            cell_size=cell_size)
    return voxel_image

def ea_render(voxel,absorption_factor,cell_size,axis=2):
    signal,density = voxel[:,:,:,:,:3],voxel[:,:,:,:,3]
    density.unsqueeze_(-1)
    density*=(absorption_factor/cell_size)
    one_minus_density = 1-density
    transmission = torch.cumprod(one_minus_density,dim=axis-4)
    
    weight = density * transmission
    weight_sum = torch.sum(weight,dim=axis-4)
    
    rendering = torch.sum(weight*signal,dim=axis-4)
    rendering = rendering/(weight_sum+1e-6)
    
    transparency = torch.prod(one_minus_density,dim=axis-4)
    alpha = 1-transparency
    
    image = torch.cat((rendering,alpha),axis=-1)
    
    return image

def diff_object_rotation_in_blender_world(voxels,rotation_angles):
    """Rotate the voxels as in blender world."""
    euler_angles = np.array([0, 0, 1], dtype=np.float32)*np.deg2rad(90)
    object_correction_matrix = rotation_matrix_3d.from_euler(euler_angles)
    
    euler_angles = np.expand_dims(rotation_angles, axis=0)
    # euler_angles = np.array([0, 1, 0], dtype=np.float32)*(-object_rotation)
    object_rotation_matrix = rotation_matrix_3d.from_euler(euler_angles)
    euler_angles_blender = np.array([1, 0, 0], dtype=np.float32)*np.deg2rad(-90)
    blender_object_correction_matrix = \
    rotation_matrix_3d.from_euler(euler_angles_blender)
    transformation_matrix = tf.matmul(tf.matmul(object_correction_matrix,
                                                object_rotation_matrix),
                                    blender_object_correction_matrix)

    return diff_transform_volume(voxels, transformation_matrix)

def diff_transform_volume(voxels, transformation_matrix,voxel_size = (128,128,128)):
    volume_sampling = helpers.sampling_points_from_3d_grid(voxel_size)
    volume_sampling = tf.matmul(transformation_matrix,
                                tf.transpose(a=volume_sampling))
    volume_sampling = tf.cast(tf.linalg.matrix_transpose(volume_sampling),
                                tf.float32)
    permuted_voxels = voxels.permute(0,4,1,2,3)
    sampling_tensor = torch.tensor(volume_sampling.numpy(),dtype=voxels.dtype).to(voxels.device)
    sampling_tensor = sampling_tensor.view(-1,128,128,128,3)
    sampling_tensor = sampling_tensor.expand(voxels.shape[0],-1,-1,-1,-1)

    output = torch.nn.functional.grid_sample(permuted_voxels,sampling_tensor)
    output = output.permute(0,2,3,4,1)
    return output

def diff_preprocess(object_voxels,rotation_angles):
    print('shapes:',object_voxels.shape,rotation_angles.shape)
    object_voxels = diff_load_voxel(object_voxels)
    interpolated_voxels = diff_estimate_ground_image(object_voxels,rotation_angles)

    ground_image, ground_alpha = \
        helpers.generate_ground_image(IMAGE_SIZE, IMAGE_SIZE, focal, principal_point,
                            camera_rotation_matrix,
                            camera_translation_vector[:, :, 0],
                            GROUND_COLOR)
    
    ground_image = torch.tensor(ground_image.numpy(),dtype=torch.float32).to(object_voxels.device)
    ground_alpha = torch.tensor(ground_alpha.numpy(),dtype=torch.float32).to(object_voxels.device)
    ground_image = ground_image.permute(0,3,1,2)
    ground_alpha = ground_alpha.permute(0,3,1,2)
    
    
    object_translation_dvr = np.array(object_translation[..., [0, 2, 1]], 
                                    dtype=np.float32)
    object_translation_dvr -= np.array([0, 0, helpers.OBJECT_BOTTOM],
                                        dtype=np.float32)
    
    rerendering = \
    diff_render_voxels_from_blender_camera(object_voxels,
                                        rotation_angles,
                                        object_translation_dvr,
                                        256, 
                                        256,
                                        focal,
                                        principal_point,
                                        camera_rotation_matrix,
                                        camera_translation_vector,
                                        absorption_factor=1.0,
                                        cell_size=1.1,
                                        depth_min=3.0,
                                        depth_max=5.0,
                                        frustum_size=(128, 128, 128))
    
    rerendering_image, rerendering_alpha = rerendering[:,:,:,:3], rerendering[:,:,:,3]

    rerendering_image=rerendering_image.permute(0,3,1,2)
    rerendering_alpha=rerendering_alpha.unsqueeze(-1).permute(0,3,1,2)
    
    rerendering_image = F.resize(rerendering_image, (256, 256))
    rerendering_alpha = F.resize(rerendering_alpha, (256, 256))
    
    BACKGROUND_COLOR = 0.784
    final_composite = BACKGROUND_COLOR*(1-rerendering_alpha)*(1-ground_alpha) + \
                    ground_image*(1-rerendering_alpha)*ground_alpha + \
                    rerendering_image*rerendering_alpha
                    
    return final_composite.to(object_voxels.device),interpolated_voxels.to(object_voxels.device)

class Preprocessor(nn.Module):
    
    def __init__(self):
        super(Preprocessor, self).__init__()
        
    def forward(self,voxels,orthogonal):
        print('module:',voxels.device)
        batch_size=voxels.shape[0]
        rotation_x = -np.deg2rad(np.random.uniform(0,360,size=(batch_size,1))).astype(np.float32)
        rotation_y = -np.deg2rad(np.random.uniform(0,360,size=(batch_size,1))).astype(np.float32)
        rotation_z = -np.deg2rad(np.random.uniform(0,360,size=(batch_size,1))).astype(np.float32)

        rotation_angles = np.concatenate([rotation_x,rotation_y,rotation_z],axis=1)


        if orthogonal:
            assert batch_size ==3
            rotations_1 = np.random.uniform(0,360,size=(batch_size,3))
            rotations_2 = rotations_1 + np.expand_dims(np.array([0,0,90]),0)
            rotations_3 = rotations_1 + np.expand_dims(np.array([0,90,0]),0)
            rotations = np.concatenate([rotations_1,rotations_2,rotations_3],axis=0)
            
        final_composite, interpolated_voxels = diff_preprocess(voxels,rotation_angles)
        
        return final_composite,interpolated_voxels
        

def save_output(output, path):
    view = 0 #@param {type:"slider", min:0, max:9, step:1}
    
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(output.squeeze().detach().cpu()*0.5+0.5)
    ax.axis('off')
    ax.set_title('NVR+ prediction')
    plt.savefig(path)
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--differentiable', action='store_true')
    args = parser.parse_args()

    path="airplane_128.npy"
    with open(path, 'rb') as f:
        voxel = np.load(f)
    torch_voxel = torch.from_numpy(voxel).to(device).float()
    torch_voxel.requires_grad=True
    
    if not args.differentiable:
        final_composite,interpolated_voxels = og_preprocess(torch_voxel)
        final_composite = torch.tensor(final_composite.numpy(),dtype=torch.float32).to(device)
    else:
        final_composite,interpolated_voxels = diff_preprocess(torch_voxel)
        final_composite = final_composite.permute(0,2,3,1)
    
    # render and save final-composite
    save_output(final_composite,"preprocess_diff=%s.png" % args.differentiable)
