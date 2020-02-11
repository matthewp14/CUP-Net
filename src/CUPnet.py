#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:05:10 2020

@author: matthewparker

NOTE: For now we are sticking to videos with dims: 30x30x30 (x,y,t) to limit toal parameters
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Multiply, Lambda
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import np_utils

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
img_rows = 28
img_cols = 28
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

def cupnet(input_shape = (30,30)):
    model = Sequential()
    
    #Trainable binary mask layer
    model.add(BinaryConv2D(1, kernel_size=input_shape, input_shape=input_shape,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv1'))
    ## Static lambda layers for streaking and integrating. 
    model.add(Lambda(streak, output_shape = streak_output_shape))
    model.add(Lambda(integrate_ims, output_shape = integrate_ims_output_shape))
    ## Flatten into vector for FCL
    model.add(Flatten()) # output shape should be 1800
    
    model.add(Dense(27000)) ## TODO: figure out what to do here about number of parameters
    model.add(Reshape((30,30,30)))
    
    ## U-Net 
    
    model.add