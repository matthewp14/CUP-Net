# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:48:50 2020

@author: f002q97
"""


import tensorflow as tf
from tensorflow.keras import * 
import cv2
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
from CUPnet import unet


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
    images = []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_vids.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")

    return images


ims = read_many_hdf5(4048)
ims = np.reshape(ims, (-1,30,32,32,1))
labels = ims


X_train, x_test, Y_train, y_test = train_test_split(ims, labels, test_size = 0.33, random_state = 42)


model = unet()
model.summary()


history = model.fit(X_train, Y_train,
          batch_size=50, epochs=10,
          verbose=2)