# Matthew Parker
# Engs 87/88 Honors Thesis
# Code to test the implementation of
# Lambda layers for CNN

import numpy as np

"""
Streak: Simulation of streak camera. 

Params: 
    input: np.array
    
Output:
    streaked np.array. new y dimension = original_y + total_images
"""
def streak(input):
    dims = np.shape(input)
    new_dims = streak_output_shape(input)
    final_streak_tensor = np.zeros(new_dims)
    sub_tensor_dims = (new_dims[1], new_dims[2], new_dims[3], new_dims[4]) #(30,60,30,1)
    vid_num = 0
    for vid in input:
        i = 0 
        sub_streak_tensor = np.zeros(sub_tensor_dims)
        for im in vid:
            sub_streak_tensor[i,i:dims[2]+i,:,:] = im
            i+=1
        final_streak_tensor[vid_num] = sub_streak_tensor
        vid_num+=1

    return final_streak_tensor

def streak_output_shape(input):
    dims = np.shape(input)
    return (dims[0],dims[1],dims[1]+dims[2],dims[3], dims[4])

"""
Integrate_ims: Simulation of CCD Imaging. Integrate along t-axis

Params:
    input: np.array
    
Output: 
    integrated np.array.
"""
def integrate_ims(input):
    output_dims = integrate_ims_output_shape(input)
    output = np.zeros(output_dims)
    print("integrated output shape " + str(np.shape(output)))
    i = 0
    for vid in input:
        for im in vid:
            output[i] += im
    return output

def integrate_ims_output_shape(input):
    dims = np.shape(input)
    return (dims[0], dims[2],dims[3],dims[4])
    
