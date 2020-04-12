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
    paddings = tf.constant([[0,0],[0,0],[0,output_shape[1]],[0,0]]) # use paddings for adding zeros
    x = tf.pad(x,paddings) # pad each frame with zeros below the data 
    streak_starter = np.zeros((output_shape[1],output_shape[2],output_shape[2])) # cant have color channel for np.fill_diagonal
    # loop and fill diagonals on streak starter for shifting later
    for i in range(output_shape[1]):
        np.fill_diagonal(streak_starter[i,i:,:],1)
    streak_starter = tf.convert_to_tensor(streak_starter,dtype="float32")
    outputs = tf.linalg.matmul(streak_starter,x) #shift multiply 
    print(np.shape(outputs))
    return outputs
    
def streak_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
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
    assert len(shape) == 4
    return tuple(shape[0],shape[2],shape[3])

    
