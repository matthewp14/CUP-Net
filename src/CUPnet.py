#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:05:10 2020

@author: matthewparker

NOTE: For now we are sticking to videos with dims: 30x30x30 (x,y,t) to limit toal parameters
"""
import tensorflow as tf
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
batch_size = 2
epochs = 20
channels = 1
img_rows = 32
img_cols = 32
filters = 32
kernel_size = (32, 32)
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

data_shape = tf.TensorShape([30,32,32,1])

def ssim_loss(y_true,y_pred):
    loss = tf.reduce_mean(tf.image.ssim(y_true,y_pred,1))
    return loss


def unet(pretrained_weights = None,input_shape = data_shape,input_size = (30,32,32,1)):
    inputs = Input(shape=data_shape,batch_size=batch_size)
    bin_conv1 = TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=input_size,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='bin_conv_1'))(inputs)
    s = Lambda(streak, output_shape = streak_output_shape)(bin_conv1)
    i = Lambda(integrate_ims, output_shape = integrate_ims_output_shape) (s)
    f = Flatten()(i)
    dense1 = Dense(30720, activation = 'relu')(f)
    resh = Reshape((30,32,32,1))(dense1)
    c1 = TimeDistributed(Conv2D(16, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (resh)
    c1 = Dropout(0.1) (c1)
    c1 = TimeDistributed(Conv2D(16, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

    c2 = TimeDistributed(Conv2D(32, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same'))(p1)
    c2 = Dropout(0.1) (c2)
    c2 = TimeDistributed(Conv2D(32, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)) )(c2)

    c3 = TimeDistributed(Conv2D(64, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(p2)
    c3 = Dropout(0.2) (c3)
    c3 = TimeDistributed(Conv2D(64, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)) )(c3)
    
    c4 = TimeDistributed(Conv2D(128, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(p3)
    c4 = Dropout(0.2) (c4)
    c4 = TimeDistributed(Conv2D(128, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(c4)
    p4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2))) (c4)

    c5 = TimeDistributed(Conv2D(256, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (p4)
    c5 = Dropout(0.3) (c5)
    c5 = TimeDistributed(Conv2D(256, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (c5)

    u6 = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))(c5)
    u6 = concatenate([u6, c4])
    c6 = TimeDistributed( Conv2D(128, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same') )(u6)
    c6 = Dropout(0.2) (c6)
    c6 = TimeDistributed(Conv2D(128, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (c6)

    u7 = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') )(c6)
    u7 = concatenate([u7, c3])
    c7 = TimeDistributed(Conv2D(64, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (u7)
    c7 = Dropout(0.2) (c7)
    c7 = TimeDistributed(Conv2D(64, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (c7)
    
    u8 = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') )(c7)
    u8 = concatenate([u8, c2])
    c8 = TimeDistributed(Conv2D(32, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (u8)
    c8 = Dropout(0.1) (c8)
    c8 = TimeDistributed(Conv2D(32, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (c8)

    u9 = TimeDistributed(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')) (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = TimeDistributed(Conv2D(16, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (u9)
    c9 = Dropout(0.1) (c9)
    c9 = TimeDistributed(Conv2D(16, (2, 2), activation='elu', kernel_initializer='he_normal', padding='same')) (c9)
    
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid')) (c9)

    model = Model(inputs = [inputs], outputs = [outputs])
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = ssim_loss, metrics = ['accuracy'])
    
    return model



if __name__=='__main__':    
    unet = unet()

    