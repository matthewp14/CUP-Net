# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:17:29 2020

@author: f002q97
"""


import sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append("../")
np.random.seed(1337)  # for reproducibility  
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
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
import scipy.io as sio




#ptm = "../../../../MATLAB/CUP Imaging with TwIST/CUP Imaging with TwIST/"
ptm = "D:/Matt/"
ball_vids = h5py.File(ptm+'full_sized_1.hdf','r')
ball_vids = np.asarray(ball_vids['data'])
ball_vids = np.transpose(ball_vids,(0,3,1,2))
ims = np.reshape(ball_vids,(100,100,256,256,1))
validate = h5py.File(ptm+'full_ball_vids.h5','r')
validate = np.asarray(validate['data'])
validate = np.transpose(validate,(0,3,1,2))
validate = validate - np.min(validate)
validate = np.reshape(validate,(100,100,256,256,1))


MX_train, MX_test, My_train, My_test = train_test_split(validate,ims, test_size = 0.3, random_state = 42)

print(np.shape(MX_test))
print(np.shape(MX_train))


reduce_lr = ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=0.5,
                              patience=20, min_lr=1e-7)
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
UNET MODEL (NO BATCH NORM)
Fixing the weights for the bin_conv1 layer as well as the dense1 layer, ie NON TRAINABLE
Feeding in weights from the forward_model above to see if that improves the results from previous session

"""

inputs = Input(shape=(100,256,256,1))
c1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (inputs)
c1 = Dropout(0.1) (c1)
c1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c1)
p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

c2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(p1)
c2 = Dropout(0.1) (c2)
c2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c2)
p2 = TimeDistributed(MaxPooling2D((2, 2)) )(c2)

c3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(p2)
c3 = Dropout(0.2) (c3)
c3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c3)
p3 = TimeDistributed(MaxPooling2D((2, 2)) )(c3)

c4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(p3)
c4 = Dropout(0.2) (c4)
c4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(c4)
p4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2))) (c4)

c5 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (p4)
c5 = Dropout(0.3) (c5)
c5 = TimeDistributed(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c5)

u6 = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same'))(c5)
u6 = concatenate([u6, c4])
c6 = TimeDistributed( Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') )(u6)
c6 = Dropout(0.2) (c6)
c6 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c6)

u7 = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') )(c6)
u7 = concatenate([u7, c3])
c7 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (u7)
c7 = Dropout(0.2) (c7)
c7 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c7)
    
u8 = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') )(c7)
u8 = concatenate([u8, c2])
c8 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (u8)
c8 = Dropout(0.1) (c8)
c8 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c8)

u9 = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')) (c8)
c9 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (u9)
c9 = Dropout(0.1) (c9)
c9 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')) (c9)
    
outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid')) (c9)

CUPNET2 = Model(inputs = [inputs], outputs = [outputs])
    
CUPNET2.compile(optimizer = Nadam(), loss = custom_loss, metrics = [ssim_loss,mse_loss])
CUPNET2.summary()


CUPNET2_history = CUPNET2.fit(MX_train, My_train,
          batch_size = 3,epochs= 100,
          verbose=1,validation_data=(MX_test,My_test),callbacks=[reduce_lr,early_stopping])