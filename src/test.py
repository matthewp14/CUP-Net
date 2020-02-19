#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:40:52 2020

@author: matthewparker
"""
'''Trains a simple binarize CNN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.98% test accuracy after 20 epochs using tensorflow backend
'''
#from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Multiply, TimeDistributed
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D

import cv2
from matplotlib import pyplot as plt
from lambda_layers import streak, integrate_ims



def binary_tanh(x):
    return binary_tanh_op(x)

import h5py
from pathlib import Path

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
    file = h5py.File(hdf5_dir / f"{num_images}_shoes.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")

    return images

ims = read_many_hdf5(88)
ims = ims.reshape(88,30,30,30,1)

print(np.shape(ims))
H = 1.
kernel_lr_multiplier = 'Glorot'

# # nn
batch_size = 50
epochs = 20
channels = 1
img_rows = 30
img_cols = 30
filters = 32
kernel_size = (30, 30)
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

# # the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(60000, 28, 28, 1)
# X_test = X_test.reshape(10000, 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
# Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1

# X_train = X_train[0:30]
# print(np.shape(X_train))

# X_train_final = np.zeros((5,30,28,28,1))




model = Sequential()

cnn = BinaryConv2D(1, kernel_size=kernel_size, input_shape=(30,img_rows, img_cols,channels),
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv1')
bk = cnn.build((30,30,30,1))
outputs, bk_temp = cnn.call(ims)

plt.figure(0)
plt.imshow(np.reshape(ims[0][0],(30,30)),cmap='gray', interpolation='nearest')
plt.figure(1)
plt.imshow(np.reshape(outputs[0][0],(30,30)),cmap='gray', interpolation='nearest')
plt.figure(2)
plt.imshow(bk_temp[0][:,:,0], interpolation='nearest')

print("outputs:"+str(np.shape(outputs)))
s_im = streak(outputs)
plt.figure(3)
plt.imshow(np.reshape(s_im[20][1],(-1,30)),interpolation='nearest')

final_ims = integrate_ims(s_im)

plt.figure(4)
plt.imshow(np.reshape(final_ims[0], (-1,30)),interpolation='nearest')