#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:21:11 2020

@author: matthewparker
"""

#from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Multiply
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D
from lambda_layers import streak, integrate_ims

import cv2
from matplotlib import pyplot as plt



def binary_tanh(x):
    return binary_tanh_op(x)


H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 50
epochs = 20
channels = 1
img_rows = 256
img_cols = 256
filters = 32
kernel_size = (256, 256)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()


#X_train = X_train.reshape(60000, 28, 28, 1)
#X_test = X_test.reshape(10000, 28, 28, 1)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
## convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
#Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1
X_train = np.zeros((80,256,256,1))
for i in range(1,81):
    im = cv2.imread('../data/'+str(i)+'.png',0)
    im = np.reshape(im,(256,256,1))
    X_train[i-1,:,:,:] = im
    

print(np.shape(X_train))


model = Sequential()

cnn = BinaryConv2D(1, kernel_size=kernel_size, input_shape=(img_rows, img_cols,channels),
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv1')
bk = cnn.build((80,256,256,1))
outputs, bk_temp = cnn.call(X_train)

print(np.shape(bk_temp))
plt.figure(0)
plt.imshow(np.reshape(X_train[0],(256,256)), interpolation='nearest',cmap='hot')
plt.figure(1)
plt.imshow(np.reshape(outputs[0],(256,256)), interpolation='nearest',cmap='hot')
plt.figure(2)
plt.imshow(bk_temp[0][:,:,0], interpolation='nearest',cmap='hot')

s_ims = streak(outputs)
final = integrate_ims(s_ims)

plt.figure(3)
plt.imshow(np.reshape(final,(-1,256)),interpolation='nearest', cmap='hot')