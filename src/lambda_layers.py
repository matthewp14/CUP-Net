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
# def streak(input):
#     dims = np.shape(input)
#     new_dims = streak_output_shape(K.shape(input))
#     final_streak_tensor = np.zeros(new_dims)
#     sub_tensor_dims = (new_dims[1], new_dims[2], new_dims[3], new_dims[4]) #(30,60,30,1)
#     vid_num = 0
#     for vid in input:
#         i = 0 
#         sub_streak_tensor = np.zeros(sub_tensor_dims)
#         for im in vid:
#             sub_streak_tensor[i,i:dims[2]+i,:,:] = im
#             i+=1
#         final_streak_tensor[vid_num] = sub_streak_tensor
#         vid_num+=1

#     return final_streak_tensor


# def streak_output_shape(input):
#     print(np.shape(input))
#     temp_tens = K.eval(input)
#     dims = K.shape(temp_tens)
#     return (dims[0],dims[1],dims[1]+dims[2],dims[3], dims[4])

@tf.function
def streak(x):
    shape = list(x)
    streak_tensor = np.zeros((shape[1],shape[1]+shape[2],shape[3],shape[4]))
    i = 0
    for im in x:
        streak_tensor[i,i:shape[2]+i,:,:] = im
        i+=1
    return streak_tensor

def streak_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] = shape[0]+shape[1]
    return tuple(shape)    


"""
Integrate_ims: Simulation of CCD Imaging. Integrate along t-axis

Params:
    input: np.array
    
Output: 
    integrated np.array.
"""
# def integrate_ims(input):
#     output_dims = integrate_ims_output_shape(input)
#     output = np.zeros(output_dims)
#     print("integrated output shape " + str(np.shape(output)))
#     i = 0
#     for vid in input:
#         for im in vid:
#             output[i] += im
#     return output

# def integrate_ims_output_shape(input):
#     dims = np.shape(input)
#     return (dims[0], dims[2],dims[3],dims[4])


def integrate_ims(x):
    return K.sum(x,axis=0)

def integrate_ims_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    return tuple(shape[1],shape[2],shape[3])
    
