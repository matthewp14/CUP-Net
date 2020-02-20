#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:05:10 2020

@author: matthewparker

NOTE: For now we are sticking to videos with dims: 30x30x30 (x,y,t) to limit toal parameters
"""
import tensorflow
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
#from tensorflow.keras.utils import np_utils

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D

import cv2
from matplotlib import pyplot as plt
from lambda_layers import streak, streak_output_shape, integrate_ims, integrate_ims_output_shape




def binary_tanh(x):
    return binary_tanh_op(x)

H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 50
epochs = 20
channels = 1
img_rows = 30
img_cols = 30
filters = 32
kernel_size = (30, 30)
pool_size = (2, 2)
hidden_units = 128
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

# def cupnet(input_shape = (30,30,30,1)):
#     model = Sequential()
    
#     #Trainable binary mask layer
#     model.add(BinaryConv2D(1, kernel_size=input_shape, input_shape=input_shape,
#                        data_format='channels_last',
#                        H=H, kernel_lr_multiplier=kernel_lr_multiplier,
#                        padding='same', use_bias=use_bias, name='conv1'))
#     ## Static lambda layers for streaking and integrating. 
#     model.add(Lambda(streak, output_shape = streak_output_shape))
#     model.add(Lambda(integrate_ims, output_shape = integrate_ims_output_shape))
#     ## Flatten into vector for FCL
#     model.add(Flatten()) # output shape should be 1800
    
#     model.add(Dense(27000,activation='relu'))
#     model.add(Reshape((30,30,30,1)))
    
#     ## U-Net 
    
#     model.add


def unet(pretrained_weights = None,input_size = (30,30,30,1)):
    inputs = Input(input_size)
    bin_conv1 = TimeDistributed(BinaryConv2D(1, kernel_size=(30,30), input_shape=input_size,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='bin_conv_1'))(inputs)
    s = Lambda(streak, output_shape = streak_output_shape)(bin_conv1)
    i = Lambda(integrate_ims, output_shape = integrate_ims_output_shape) (s)
    f = Flatten()(i)
    dense1 = Dense(27000, activatiion = 'relu')(f)
    resh = Reshape((30,30,30,1))(dense1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(resh)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model



unet = unet()
    