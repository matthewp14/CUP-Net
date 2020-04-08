"""
Created on Wed Feb 26 17:01:56 2020

"""

""" IMPORTS """
import sys
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.constraints import *
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D

import h5py
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from lambda_layers import *
from binary_ops import *

""" FUNCTIION AND VARIABLE DEFINITIONS """
def binary_tanh(x):
    return binary_tanh_op(x)

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
    #return streak_tensor
    return np.sum(streak_tensor,axis=1)

def mask(val,ims,mask):
    for i in range(np.shape(val)[0]):
        for j in range(np.shape(val)[1]):
            val[i,j,:,:] = ims[i,j,:,:] * mask
    return val



ims = read_many_hdf5(859)
# ims = np.ones((3943,30,32,32,1))
ims = np.reshape(ims, (-1,30,32,32,1))
ims = ims[:840]
# temp = np.zeros((1,32,32,1))
validate2 = np.zeros((840,30,32,32,1))
bk_temp = np.random.randint(0,2,(1,32,32,1))
validate2 = mask(validate2,ims,bk_temp)
ims2 = np_streak(validate2)


validate = ims

validate = validate / 255
ims2 = ims2 /255
ims = ims/255
#X_train, X_test, y_train, y_test = train_test_split(ims, validate, test_size=(1/3), random_state=42)
X_train, X_test, y_train, y_test = train_test_split(ims2, validate, test_size=(1/3), random_state=42)

MX_train, MX_test, My_train, My_test = train_test_split(ims,ims, test_size = 1/3, random_state = 42)

print(np.shape(X_test))
print(np.shape(X_train))

reduce_lr = ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=0.5,
                              patience=50, min_lr=0.000001)
early_stopping = EarlyStopping(patience=90,verbose=1,restore_best_weights=True)


def custom_loss(y_true, y_pred):

  ssim_loss = (1.0-tf.image.ssim(y_true,y_pred,1))/2.0
  mse_loss = K.mean(K.square(y_pred-y_true))
  #mse_loss = tf.keras.losses.mean_squared_error(y_true,y_pred)

  ssim_loss = 0.5*ssim_loss
  mse_loss = 0.5*mse_loss

  return ssim_loss + mse_loss

def ssim_loss(y_true,y_pred):
    return (1.0-tf.image.ssim(y_true,y_pred,1))/2.0

def mse_loss(y_true,y_pred):
    return K.mean(K.square(y_pred-y_true))
"""
Fine Tuning Parameters: 
Ratio between MSE and SSIM 
"""

ssim_alpha = 0.1
ssim_beta = 0.9
def variable_custom_loss(y_true, y_pred):
    global ssim_alpha, ssim_beta
    ssim_loss = (1.0-tf.image.ssim(y_true,y_pred,1))/2.0
    mse_loss = K.mean(K.square(y_pred-y_true))
    #mse_loss = tf.keras.losses.mean_squared_error(y_true,y_pred)

    ssim_loss = ssim_alpha*ssim_loss
    mse_loss = ssim_beta*mse_loss
    return ssim_loss + mse_loss


alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
beta = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

mse_losses = []
ssim_losses = []
mse_sd = []
ssim_sd = []

for i in range(9):
    ssim_alpha = alpha[i]
    ssim_beta = beta[i]
    inner_mse_losses=[]
    inner_ssim_losses=[]
    for j in range(1):
        MX_train, MX_test, My_train, My_test = train_test_split(ims,ims, test_size = 1/3)
        forward_model = Sequential()
        forward_model.add(Input(shape=(30,32,32,1),batch_size = 40))
        forward_model.add(TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=(30,32,32,1),
                           data_format='channels_last',
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                           padding='same', use_bias=use_bias, name='bin_conv_1')))
        forward_model.add(Lambda(streak,output_shape=streak_output_shape))
        forward_model.add(Lambda(integrate_ims, output_shape = integrate_ims_output_shape))
        forward_model.add(Flatten())
        forward_model.add(Dense(30720, activation = 'relu'))
        forward_model.add(Reshape((30,32,32,1)))
        forward_model.compile(optimizer = Nadam(0.0001), loss = variable_custom_loss, metrics = ['mean_squared_error',mse_loss,ssim_loss])
        forward_model.summary()
        forward_history = forward_model.fit(MX_train, My_train,
              batch_size = 40,epochs= 1,
              verbose=2,validation_data=(MX_test,My_test),callbacks=[reduce_lr])
        evals = forward_model.evaluate(MX_test,My_test)
        inner_mse_losses.append(evals[1])
        inner_ssim_losses.append(evals[2])
        K.clear_session()
    mse_losses.append(np.mean(inner_mse_losses))
    ssim_losses.append(np.mean(inner_ssim_losses))
    mse_sd.append(np.std(inner_mse_losses))
    ssim_sd.append(np.std(inner_ssim_losses))


with open("training_logs.txt",'w') as file:
    for i in range(10):
        file.write("[ssim,mse]: " + "["+str(alpha[i])+","+str(beta[i])+"]"+ "MSE LOSS: " + str(mse_losses[i]) + " +/- " + str(mse_sd[i]) + + " SSIM LOSS: " + str(ssim_losses[i]) + " +/- " + str(ssim_sd[i]) + "\n")
    
    
    
