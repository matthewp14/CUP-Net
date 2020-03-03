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
    paddings = tf.constant([[0,0],[0,0],[0,output_shape[1]],[0,0],[0,0]])
    x = tf.pad(x,paddings) # pad each frame with zeros below the data 
    streak_starter = np.zeros((output_shape[1],output_shape[2],output_shape[2])) # cant have color channel for np.fill_diagonal
    
    for i in range(output_shape[1]):
        np.fill_diagonal(streak_starter[i,i:,:],1)
    streak_starter = tf.convert_to_tensor(streak_starter,dtype="float32")
    x = tf.reshape(x,(np.shape(x)[:-1]))
    outputs = tf.linalg.matmul(streak_starter,x)
    return tf.reshape(outputs, output_shape)
    
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



# @tf.function
# def streak(x):
#     output_shape = streak_output_shape(np.shape(x))
#     print(type(output_shape))
#     # paddings = tf.constant([[0,0],[0,0],[0,output_shape[1]],[0,0],[0,0]])
#     paddings = tf.constant([[0,0],[0,0],[0,output_shape[1]],[0,0]])
#     x = tf.pad(x,paddings) 
#     streak_starter = np.zeros((output_shape[1],output_shape[2],output_shape[2])) # cant have color channel for np.fill_diagonal
#     for i in range(output_shape[1]):
#         np.fill_diagonal(streak_starter[i,i:,:],1)
#     # streak_starter = np.reshape(streak_starter, (output_shape[1],output_shape[2],output_shape[2],1))
#     # streak = streak_starter
#     streak_starter = tf.convert_to_tensor(streak_starter,dtype="float32")
#     # x = tf.reshape(x,(np.shape(x)[:-1]))
#     # reshape_dims = list(output_shape[:-1])
#     # reshape_dims[0] = tf.shape(x)[0].value
#     # reshape_dims = tf.TensorShape(tuple(reshape_dims))
#     # x = tf.reshape(x,[-1,30,62,32])
#     # x = tf.reshape(x,tf.constant(reshape_dims))
#     # x = tf.reshape(x,(None,input_dims[1],input_dims[2],input_dims[3]))
#     outputs = tf.linalg.matmul(streak_starter,x)
#     return outputs
    
# def streak_output_shape(input_shape):
#     shape = list(input_shape)
#     assert len(shape) == 4
#     shape[2] = shape[1] + shape[2]
#     return tuple(shape)

    
