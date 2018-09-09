import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

def load_cifar10_data(img_rows, img_cols, num_obs_train=None, num_obs_valid=None, test_path=None):
    
    num_classes = 10
    
    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    
    if num_obs_valid is not None:
        X_valid, Y_valid = X_valid[:num_obs_valid], Y_valid[:num_obs_valid]
        
    if num_obs_train is not None:
        X_train, Y_train = X_train[:num_obs_train], Y_train[:num_obs_train]
        
    if test_path is not None:
        test = np.load(test_path)
        
    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:,:,:,:]])
        if test_path is not None:
            test = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in test[:,:,:,:]])        
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:,:,:,:]])
        if test_path is not None:
            test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in test[:,:,:,:]])
            
    Y_train = np_utils.to_categorical(Y_train[:], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:], num_classes)
    
    if test_path is not None: 
        return X_train, Y_train, X_valid, Y_valid, test
    else: 
        return X_train, Y_train, X_valid, Y_valid
