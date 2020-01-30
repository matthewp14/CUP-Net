# Matthew Parker
# Engs 87/88 Honors Thesis
# Code to test the implementation of

from __future__ import print_function
import keras
from keras import layers
from keras.layers import Lambda
from keras import backend as K
import numpy as np
from keras.datasets import mnist

import cv2
from matplotlib import pyplot as plt


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

def integrate_ims(input):
    output = input[0]
    for i in range(1, np.shape(input)[0]):
        output+= input[i]
    return output

def integrate_ims_shape(input):
    dims = np.shape(input)
    return (1, dims[1],dims[2],dims[3])
    
    
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train[0:30]

s_t = streak(X_train)

print(np.shape(s_t))
plt.figure(0)
plt.imshow(np.reshape(s_t[5],(-1,28)), interpolation='nearest')
plt.figure(1)
plt.imshow(np.reshape(s_t[0],(-1,28)), interpolation='nearest')



s_t_final=integrate_ims(s_t)

plt.figure(3)
plt.imshow(np.reshape(s_t_final,(-1,28)),interpolation='nearest')