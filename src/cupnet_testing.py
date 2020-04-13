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
from numba import cuda

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


ims = read_many_hdf5("training")
ims = np.reshape(ims, (-1,30,32,32,1))

validate = ims

validate = validate / 255
ims = ims /255

MX_train, MX_test, My_train, My_test = train_test_split(ims,ims, test_size = 1/3, random_state = 42)


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

forward_model = Sequential()
forward_model.add(Input(shape=(30,32,32,1),batch_size = 32))
forward_model.add(TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=(30,32,32,1),
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='bin_conv_1')))
forward_model.add(Reshape((30,32,32)))
forward_model.add(Lambda(streak,output_shape=streak_output_shape))
forward_model.add(Lambda(integrate_ims, output_shape = integrate_ims_output_shape))
forward_model.add(Flatten())
forward_model.add(Dense(30720, activation = 'relu'))
forward_model.add(Reshape((30,32,32,1)))
forward_model.compile(optimizer = Nadam(0.0001), loss = custom_loss, metrics = ['mean_squared_error',mse_loss])

forward_model.load_weights('../data/model_stuff/forward_weights_4_12.h5')
binary_weights = forward_model.layers[0].get_weights()
inverse_weights = forward_model.layers[5].get_weights()

for i in range(5,9):
    ssim_alpha = alpha[i]
    ssim_beta = beta[i]
    inner_mse_losses=[]
    inner_ssim_losses=[]
    for j in range(2):
        MX_train, MX_test, My_train, My_test = train_test_split(ims,ims, test_size = 1/3)
        """ 
        UNET MODEL
        Fixing the weights for the bin_conv1 layer as well as the dense1 layer, ie NON TRAINABLE
        Feeding in weights from the forward_model above to see if that improves the results from previous session

        """
        inputs = Input(shape=(30,32,32,1))
        bin_conv1 = TimeDistributed(BinaryConv2D(1, kernel_size=(32,32), input_shape=(30,32,32,1),
                               data_format='channels_last',
                               H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                               padding='same', use_bias=use_bias, name='bin_conv_1',trainable=False))(inputs)
        resh1 = Reshape((30,32,32))(bin_conv1)
        streak1 = Lambda(streak, output_shape = streak_output_shape)(resh1)
        integrate1 = Lambda(integrate_ims, output_shape = integrate_ims_output_shape) (streak1)
        f = Flatten()(integrate1)
        dense1 = Dense(30720, activation = 'relu',trainable=False)(f)
        resh2 = Reshape((30,32,32,1))(dense1)
        c1 = TimeDistributed(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (resh2)
        c1 = Dropout(0.1) (c1)
        c1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c1)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

        c2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(p1)
        c2 = Dropout(0.1) (c2)
        c2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c2)
        p2 = TimeDistributed(MaxPooling2D((2, 2)) )(c2)

        c3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(p2)
        c3 = Dropout(0.2) (c3)
        c3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c3)
        p3 = TimeDistributed(MaxPooling2D((2, 2)) )(c3)
            
        c4 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(p3)
        c4 = Dropout(0.2) (c4)
        c4 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c4)
        p4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2))) (c4)

        c5 = TimeDistributed(Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')) (p4)
        c5 = Dropout(0.3) (c5)
        c5 = TimeDistributed(Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')) (c5)

        u6 = TimeDistributed(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))(c5)
        u6 = concatenate([u6, c4])
        c6 = TimeDistributed( Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(u6)
        c6 = Dropout(0.2) (c6)
        c6 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c6)

        u7 = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') )(c6)
        u7 = concatenate([u7, c3])
        c7 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (u7)
        c7 = Dropout(0.2) (c7)
        c7 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c7)
            
        u8 = TimeDistributed(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') )(c7)
        u8 = concatenate([u8, c2])
        c8 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (u8)
        c8 = Dropout(0.1) (c8)
        c8 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c8)

        u9 = TimeDistributed(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')) (c8)
        c9 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (u9)
        c9 = Dropout(0.1) (c9)
        c9 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c9)
            
        outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid')) (c9)

        CUPNET = Model(inputs = [inputs], outputs = [outputs])
            
        CUPNET.compile(optimizer = Nadam(), loss = variable_custom_loss, metrics = [mse_loss,ssim_loss])
        CUPNET.layers[1].set_weights(binary_weights)
        CUPNET.layers[6].set_weights(inverse_weights)

        CUPNET_history = CUPNET.fit(MX_train, My_train,
              batch_size = 32,epochs= 100,
              verbose=2,validation_data=(MX_test,My_test),callbacks=[reduce_lr])

        CUPNET.save_weights(f'../data/model_stuff/cupnet_weights/{j}_{ssim_alpha}_cupnet_4_12.h5')
        evals = CUPNET.evaluate(MX_test,My_test)
        inner_mse_losses.append(evals[1])
        inner_ssim_losses.append(evals[2])
        memory = cuda.current_context().get_memory_info()
        print('1 Memory Status:', memory[0], 'free out of', memory[1], ',', (int(memory[0])/int(memory[1])*100), '% free')
        K.clear_session()
        memory = cuda.current_context().get_memory_info()
        print('2 Memory Status:', memory[0], 'free out of', memory[1], ',', (int(memory[0])/int(memory[1])*100), '% free')
        
    mse_losses.append(inner_mse_losses)
    ssim_losses.append(inner_ssim_losses)

    with open("cupnet_training_logs.txt",'a') as file:
        #msg = "[ssim,mse]: " + "["+str(alpha[i])+","+str(beta[i])+"]"+ " MSE LOSS: " + str(mse_losses[i]) + " +/- " + str(mse_sd[i]) + " SSIM LOSS: " + str(ssim_losses[i]) + " +/- " + str(ssim_sd[i]) + "\n"
        #msg = "[ssim,mse]: " + "["+str(alpha[i])+","+str(beta[i])+"]"+ " MSE LOSS: " + str(np.mean(inner_mse_losses)) + " +/- " + str(np.std(inner_mse_losses)) + " SSIM LOSS: " + str(np.mean(inner_ssim_losses)) + " +/- " + str(np.std(inner_ssim_losses)) + "\n"
        msg = f'[ssim,mse]: {alpha[i]} , {beta[i]} MSE LOSS: {np.mean(inner_mse_losses)} +/- {np.std(inner_mse_losses)} SSIM LOSS: {np.mean(inner_ssim_losses)} +/- {np.std(inner_ssim_losses)} \n'
        file.write(msg)
            
with open("cupnet_training_logs.txt",'a') as file:
    msg = "\n ALL HISTORY: \n" + "SSIM LOSSES: " + str(ssim_losses) + "\nMSE LOSSES: " + str(mse_losses)
    file.write(msg)
