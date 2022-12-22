import torch
import numpy as np

import tensorflow as tf

from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer import models
from tensorflow_graphics.rendering.volumetric import visual_hull

from preprocess import diff_preprocess

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
    
    upstream_grad = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                        name='upstream_grad')
    
    model = models.neural_voxel_renderer_plus(vol_placeholder,
                                                rerender_placeholder,
                                                light_placeholder)
    predicted_image_logits, = model.outputs
    
    adjusted = tf.reduce_sum(predicted_image_logits*upstream_grad)
    vol_gradients = tf.gradients(adjusted,vol_placeholder)
    rerender_gradients = tf.gradients(adjusted,rerender_placeholder)
    
    saver = tf.compat.v1.train.Saver()
    
def render_tf_forward(final_composite,interpolated_voxels):
    # tf.compat.v1.reset_default_graph()
    batch_size = interpolated_voxels.shape[0]
    a = interpolated_voxels.cpu().numpy()
    b = final_composite.cpu().numpy()*2.-1 #TODO account for this
    c = light_position
    d = np.zeros((batch_size,256,256,3))
    with tf.compat.v1.Session(graph=g) as sess:
        saver.restore(sess, latest_checkpoint)
        feed_dict = {vol_placeholder: a,
                    rerender_placeholder: b,
                    light_placeholder: c,
                    upstream_grad: d}
        predictions = sess.run(predicted_image_logits, feed_dict)

    return predictions

def render_tf_backward(final_composite,interpolated_voxels,upstream_gradient):
    latest_checkpoint = '/tmp/checkpoint/model.ckpt-126650'
    
    a = interpolated_voxels.cpu().numpy()
    b = final_composite.cpu().numpy()*2.-1
    c = light_position
    d = upstream_gradient.cpu().numpy()
    with tf.compat.v1.Session(graph=g) as sess:
        saver.restore(sess, latest_checkpoint)
        feed_dict = {vol_placeholder: a,
                    rerender_placeholder: b,
                    light_placeholder: c,
                    upstream_grad: d}
        a_deriv = sess.run(rerender_gradients,feed_dict)
        b_deriv = sess.run(vol_gradients,feed_dict)
    return a_deriv[0],b_deriv[0]

class NVR(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, final_composite,interpolated_voxels):
        ctx.save_for_backward(final_composite,interpolated_voxels)

        predictions = render_tf_forward(final_composite,interpolated_voxels)
        #gives nice output if you use matplotlib.pyplot
        torch_predictions= torch.from_numpy(predictions).float()
        print('input sizes:')
        print(final_composite.shape)
        print(interpolated_voxels.shape)
        return torch_predictions
    
    @staticmethod
    def backward(ctx, grad_output):
        final_composite,interpolated_voxels = ctx.saved_tensors
        d_composite,d_interpolated = render_tf_backward(final_composite,interpolated_voxels,grad_output)
        print('backward sizes:')
        print(d_composite.shape)
        print(d_interpolated.shape)
        import pdb;pdb.set_trace()
        return torch.from_numpy(d_composite).to('cuda:0'),torch.from_numpy(d_interpolated).to('cuda:0')


class NVR_Renderer:
    
    def __init__(self):
        pass
    
    def render(self,voxels):
        #TODO randomize light and camera angles
        final_composite,interpolated_voxels = diff_preprocess(voxels)
        final_composite.retain_grad=True
        interpolated_voxels.retain_grad=True
        output=NVR.apply(final_composite,interpolated_voxels)
        output.retain_grad=True
        permuted_output=output.permute(0,3,1,2)
        permuted_output.retain_grad=True
        return permuted_output
        
    def render_backward(self,voxels,upstream_gradient):
        pass
        
if __name__=="__main__":
    path="airplane_128.npy"
    with open(path, 'rb') as f:
        voxel = np.load(f)
    voxel = torch.from_numpy(voxel).float().to('cuda:0')
    voxel.requires_grad=True
    
    renderer = NVR_Renderer()
    output=renderer.render(voxel)
    print('hello')
    
    path="airplane_128.npy"
    with open(path, 'rb') as f:
        new_voxel = np.load(f)
    new_voxel = torch.from_numpy(new_voxel).float().to('cuda:0')
    new_voxel.requires_grad=True
    
    output=renderer.render(new_voxel)
    print('hello')
    
    output.sum().backward()
    
    print(voxel.grad)