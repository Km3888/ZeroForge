import torch
import numpy as np

import tensorflow as tf

from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer import models
from tensorflow_graphics.rendering.volumetric import visual_hull

camera_rotation_matrix= np.array([[ 9.9997330e-01,  7.3080887e-03,  8.9461202e-11],\
    [ 4.9256836e-03, -6.7398632e-01, -7.3872751e-01],\
    [-5.3986851e-03,  7.3870778e-01, -6.7400432e-01]]).astype(np.float32)
camera_translation_vector = np.array([[5.2963998e-09],\
    [5.3759331e-01],\
    [4.2457557e+00]]).astype(np.float32)
focal = np.array([284.44446, 284.44446]).astype(np.float32)
principal_point = np.array([128., 128.]).astype(np.float32)
light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773 ]).astype(np.float32)
object_rotation = np.array([139.]).astype(np.float32)
object_translation = np.array([-0.39401,  0.,  0.4]).astype(np.float32)
object_elevation = np.array([47.62312]).astype(np.float32)

camera_rotation_matrix=np.expand_dims(camera_rotation_matrix,axis=(0))
camera_translation_vector=np.expand_dims(camera_translation_vector,axis=(0)).astype(np.float32)
light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)

object_translation=np.expand_dims(object_translation,axis=(0,1))
object_rotation=np.expand_dims(object_rotation,axis=(0))
object_elevation=np.expand_dims(object_elevation,axis=(0))

#convert object arrays to float32 datatype
object_translation=object_translation.astype(np.float32)
object_rotation=object_rotation.astype(np.float32)
object_elevation=object_elevation.astype(np.float32)


VOXEL_SIZE = 128
IMAGE_SIZE = 256
GROUND_COLOR = np.array((136., 162, 199))/255.
BLENDER_SCALE = 2
DIAMETER = 4.2  # The voxel area in world coordinates

def estimate_ground_image(object_voxels):
    
    object_rotation_v = object_rotation
    object_translation_v = object_translation[:, 0, [1, 0, 2]]*BLENDER_SCALE
    object_elevation_v = object_elevation

    ground_occupancy = np.zeros((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1),
                                dtype=np.float32)
    ground_occupancy[-2, 1:-2, 1:-2, 0] = 1
    print('ground occupancy:',ground_occupancy.shape)
    ground_voxel_color = np.ones((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 3), 
                                dtype=np.float32)*\
                        np.array(GROUND_COLOR, dtype=np.float32)
    ground_voxel_color = np.concatenate([ground_voxel_color, ground_occupancy],
                                        axis=-1)

    #print('ground_voxel_color:',ground_voxel_color.shape)

    scene_voxels = object_voxels*(1-ground_occupancy) + \
                    ground_voxel_color*ground_occupancy



    euler_angles_x = np.deg2rad(180-object_rotation_v)*np.array([1, 0, 0],
                                                                dtype=np.float32)
    euler_angles_y = np.deg2rad(90-object_elevation_v)*np.array([0, 1, 0],
                                                                dtype=np.float32)
    translation_vector = (object_translation_v/(DIAMETER*0.5))

    interpolated_voxels = helpers.object_to_world(scene_voxels,
                                                euler_angles_x,
                                                euler_angles_y,
                                                translation_vector)
    #print('interpolated:',interpolated_voxels.shape)

    color_input, alpha_input = tf.split(interpolated_voxels, [3, 1], axis=-1)
    voxel_img = visual_hull.render(color_input*alpha_input)
    return interpolated_voxels

def process_voxel(voxel):
    voxel = voxel.detach().cpu().numpy()
    edited_voxel = np.transpose(voxel,(1,0,2)) 
    edited_voxel = np.flip(edited_voxel,0)
    
    edited_voxel = edited_voxel.reshape((1,128,128,128,1))
    expanded = np.repeat(edited_voxel,4,axis=4)/3
    expanded[:,:,:,:,-1]=3*expanded[:,:,:,:,0]
    
    return expanded

def preprocess(object_voxels):
    object_voxels = process_voxel(object_voxels)
        
    interpolated_voxels = estimate_ground_image(object_voxels)
    
    ground_image, ground_alpha = \
        helpers.generate_ground_image(IMAGE_SIZE, IMAGE_SIZE, focal, principal_point,
                            camera_rotation_matrix,
                            camera_translation_vector[:, :, 0],
                            GROUND_COLOR)
    object_rotation_dvr = np.array(np.deg2rad(object_rotation),
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


def render_tf_forward(object_voxels):
    
    final_composite,interpolated_voxels = preprocess(object_voxels)
    latest_checkpoint = '/tmp/checkpoint/model.ckpt-126650'

    tf.compat.v1.reset_default_graph()
    g = tf.compat.v1.Graph()
    with g.as_default():
        vol_placeholder = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 4],
                                            name='input_voxels')
        rerender_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                                name='rerender')
        light_placeholder = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, 3],
                                            name='input_light')
        model = models.neural_voxel_renderer_plus(vol_placeholder,
                                                    rerender_placeholder,
                                                    light_placeholder)
        predicted_image_logits, = model.outputs
        saver = tf.compat.v1.train.Saver()

    a = interpolated_voxels.numpy()
    b = final_composite.numpy()*2.-1
    c = light_position
    with tf.compat.v1.Session(graph=g) as sess:
        saver.restore(sess, latest_checkpoint)
        feed_dict = {vol_placeholder: a,
                    rerender_placeholder: b,
                    light_placeholder: c}
        predictions = sess.run(predicted_image_logits, feed_dict)

    return predictions

def render_tf_backward(object_voxels,upstream_gradient):
    
    final_composite,interpolated_voxels = preprocess(object_voxels)
    latest_checkpoint = '/tmp/checkpoint/model.ckpt-126650'

    tf.compat.v1.reset_default_graph()
    g = tf.compat.v1.Graph()
    with g.as_default():
        vol_placeholder = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 4],
                                            name='input_voxels')
        rerender_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                                name='rerender')
        light_placeholder = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, 3],
                                            name='input_light')
        model = models.neural_voxel_renderer_plus(vol_placeholder,
                                                    rerender_placeholder,
                                                    light_placeholder)
        predicted_image_logits, = model.outputs
                
        upstream = tf.constant(upstream_gradient)
        adjusted = tf.reduce_sum(predicted_image_logits*upstream)
        gradients = tf.gradients(adjusted,vol_placeholder)

        saver = tf.compat.v1.train.Saver()

    a = interpolated_voxels.numpy()
    b = final_composite.numpy()*2.-1
    c = light_position
    with tf.compat.v1.Session(graph=g) as sess:
        saver.restore(sess, latest_checkpoint)
        feed_dict = {vol_placeholder: a,
                    rerender_placeholder: b,
                    light_placeholder: c}
        predictions = sess.run(predicted_image_logits, feed_dict)
        grad = sess.run(gradients, feed_dict)
        derivs = sess.run(gradients,feed_dict)
    return derivs[0]

class NVR(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, object_voxels):
        ctx.save_for_backward(object_voxels)

        predictions = render_tf_forward(object_voxels)
        #gives nice output if you use matplotlib.pyplot
        torch_predictions= torch.from_numpy(predictions).float()
        
        return torch_predictions
    
    @staticmethod
    def backward(ctx, grad_output):
        import pdb; pdb.set_trace()
        voxel_input = ctx.saved_tensors[0]
        grad = render_tf_backward(voxel_input,grad_output.numpy())
        return torch.from_numpy(grad)
    
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
