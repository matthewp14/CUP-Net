# Matthew Parker
# Engs 87/88 Honors Thesis
# Code to test the implementation of
# Lambda layers for CNN

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

"""
Streak: Simulation of streak camera. 

Params: 
    input: np.array
    
Output:
    streaked np.array. new y dimension = original_y + total_images
"""

@tf.function
def streak(x):
    output_shape = streak_output_shape(np.shape(x))
    streak_tensor = np.zeros(output_shape)
    streak_tensor = tf.Variable(streak_tensor,trainable=False,dtype ="float32")
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            im = x[i,j,:,:,:]
            streak_tensor[i,j,j:(output_shape[3]+j),:,:].assign(im)
    return streak_tensor
    
def streak_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 5
    shape[2] = shape[1] + shape[2]
    return tuple(shape)


"""
Integrate_ims: Simulation of CCD Imaging. Integrate along t-axis

Params:
    input: np.array
    
Output: 
    integrated np.array.
"""


def integrate_ims(x):
    return K.sum(x,axis=1)

def integrate_ims_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 5
    return tuple(shape[0],shape[2],shape[3],shape[4])
    
