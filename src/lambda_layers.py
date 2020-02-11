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
    streak_tensor = np.zeros(new_dims)
    print(np.shape(streak_tensor))
    i = 0 
    for im in input:
        streak_tensor[i,i:dims[1]+i,:,:] = im
        i+=1
    return streak_tensor

def streak_output_shape(input):
    dims = np.shape(input)
    return (dims[0],dims[0]+dims[1],dims[2], dims[3])

"""
Integrate_ims: Simulation of CCD Imaging. Integrate along t-axis

Params:
    input: np.array
    
Output: 
    integrated np.array. Single image.
"""
def integrate_ims(input):
    output = input[0]
    for i in range(1, np.shape(input)[0]):
        output+= input[i]
    return output

def integrate_ims_output_shape(input):
    dims = np.shape(input)
    return (1, dims[1],dims[2],dims[3])
    
