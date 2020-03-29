#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:01:56 2020

"""

import numpy as np
np.random.seed(1337)  # for reproducibility

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D

import cv2
import matplotlib.pyplot as plt
from lambda_layers import *


def binary_tanh(x):
    return binary_tanh_op(x)

import h5py
from pathlib import Path
H = 1.
kernel_lr_multiplier = 'Glorot'

# # nn
batch_size = 50
epochs = 20
channels = 1
img_rows = 30
img_cols = 30
filters = 32
kernel_size = (32, 32)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False

# # learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# # BN
epsilon = 1e-6
momentum = 0.9

# # dropout
p1 = 0.25
p2 = 0.5

hdf5_dir = Path("../data/hdf5/")

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images= []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_vids.h5", "r+")

    images = np.array(file["/images"]).astype("float32")

    return images

def np_streak(x):
    input_dims = np.shape(x)
    output_shape = (input_dims[0],input_dims[1],input_dims[1]+input_dims[2],input_dims[3],input_dims[4])
    streak_tensor = np.zeros(output_shape)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            streak_tensor[i,j,j:(output_shape[3]+j),:,:] = x[i,j,:,:,:]
    return streak_tensor
    # return np.sum(streak_tensor,axis=1)

def mask(val,ims,mask):
    for i in range(np.shape(val)[0]):
        for j in range(np.shape(val)[1]):
            val[i,j,:,:] = ims[i,j,:,:] * mask
    return val

ims = read_many_hdf5(3943)
# ims = np.ones((3943,30,32,32,1))
ims = np.reshape(ims, (-1,30,32,32,1))
ims = ims[:3900]
# temp = np.zeros((1,32,32,1))


bk_temp = np.random.randint(0,2,(1,32,32,1))

validate = np.zeros((3900,30,32,32,1))
validate = mask(validate,ims,bk_temp)
validate2  = validate
validate2 = np_streak(validate)
validate3 = np.sum(validate2, axis=1)


model1 = Sequential()

model1.add(Input(shape=(30,32,32,1),batch_size = 100))

model1.add(TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=(30,32,32,1),
                        data_format='channels_last',
                        H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                        padding='same', use_bias=use_bias, name='bin_conv_1')))
model1.compile(optimizer = Adam(lr = 1), loss = 'mean_squared_error', metrics = ['mse'])
model1.summary()
history1 = model1.fit(ims, validate,
          batch_size = 100,epochs= 5,
          verbose=2)

model2 = Sequential()

model2.add(Input(shape=(30,32,32,1),batch_size = 100))

model2.add(TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=(30,32,32,1),
                        data_format='channels_last',
                        H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                        padding='same', use_bias=use_bias, name='bin_conv_1')))
model2.add(Lambda(streak, output_shape = streak_output_shape))
model2.compile(optimizer = Adam(lr = 1), loss = 'mean_squared_error', metrics = ['mse'])
model2.summary()
history2 = model2.fit(ims, validate2,
          batch_size = 100,epochs= 5,
          verbose=2)
    

""" THIS ONE DOES NOT """
model3 = Sequential()

model3.add(Input(shape=(30,32,32,1),batch_size = 100))

model3.add(TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=(30,32,32,1),
                        data_format='channels_last',
                        H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                        padding='same', use_bias=use_bias, name='bin_conv_1')))
model3.add(Lambda(streak, output_shape = streak_output_shape))
model3.add(Lambda(integrate_ims, output_shape = integrate_ims_output_shape,trainable=False))
model3.compile(optimizer = Adam(lr = 1), loss = 'mean_squared_error', metrics = ['mse'])
model3.summary()
history3 = model3.fit(ims, validate3,
          batch_size = 100,epochs= 20,
          verbose=2)



