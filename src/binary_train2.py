"""
Created on Wed Feb 26 17:01:56 2020

"""
import numpy as np
np.random.seed(1337)

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
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

import cv2
import matplotlib.pyplot as plt
from lambda_layers import *


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


ACCURACY_THRESHOLD = 0.035

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss') < ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True

# Instantiate a callback object
threshold = myCallback()

### 3943
# ims = read_many_hdf5(3943)
# # ims = np.ones((3943,30,32,32,1))
# ims = np.reshape(ims, (-1,30,32,32,1))
# ims = ims[:3900]
# # temp = np.zeros((1,32,32,1))
# validate2 = np.zeros((3900,30,32,32,1))
# bk_temp = np.random.randint(0,2,(1,32,32,1))
# validate2 = mask(validate2,ims,bk_temp)
# ims2 = np_streak(validate2)

#### 1515
ims = read_many_hdf5(1516)
# ims = np.ones((3943,30,32,32,1))
ims = np.reshape(ims, (-1,30,32,32,1))
ims = ims[:1500]
# temp = np.zeros((1,32,32,1))
validate2 = np.zeros((1500,30,32,32,1))
bk_temp = np.random.randint(0,2,(1,32,32,1))
validate2 = mask(validate2,ims,bk_temp)
ims2 = np_streak(validate2)

validate = ims

validate = validate / 255
ims2 = ims2 /255
ims = ims/255
#X_train, X_test, y_train, y_test = train_test_split(ims, validate, test_size=(1/3), random_state=42)
X_train, X_test, y_train, y_test = train_test_split(ims2, validate, test_size=(1/3), random_state=42)

print(np.shape(X_test))
print(np.shape(X_train))

reduce_lr = ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=0.2,
                              patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(patience=10,verbose=1,restore_best_weights=True)

model2 = Sequential()
model2.add(Input(shape=(62,32,1),batch_size = 100))
model2.add(Flatten())
model2.add(Dense(30720, activation = 'relu'))
model2.add(Reshape((30,32,32,1)))
model2.compile(optimizer = Nadam(0.0001), loss = 'mean_squared_error', metrics = ['mse'])
model2.summary()
history2 = model2.fit(X_train, y_train,
          batch_size = 100,epochs= 50,
          verbose=2,validation_data=(X_test,y_test),callbacks=[early_stopping,threshold])

