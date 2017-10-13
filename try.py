import os
import sys
import random

import numpy as np
import pandas as pd
from skimage import measure

import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize


from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import regularizers

from numpy import logical_and as AND
from numpy import logical_not as NOT

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'Data/stage1_train/'
TEST_PATH = 'Data/stage1_test/'
SAVE_LOAD = False

reg = 0.0001
seed = 42
random.seed = seed
np.random.seed = seed

def GetData():
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    
    
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    
    
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    
    print('Done!')
    return X_train, Y_train, X_test, sizes_test
    

# Define IoU metric
def MeanIoU(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
    
'''Build U-Net model'''
def GetUNetModel(reg):
    r = regularizers.l2(reg)
    
    # (128, 128, 3)
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)
    # (128, 128, 3) -> (128, 128, 16)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (s)
    c1 = Dropout(0.1) (c1)
    # (128, 128, 16) -> (128, 128, 16)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c1)
    # (128, 128, 16) -> (64, 64, 16)
    p1 = MaxPooling2D((2, 2)) (c1)
    # (64, 64, 16) -> (64, 64, 32)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p1)
    c2 = Dropout(0.1) (c2)
    # (64, 64, 32) -> (64, 64, 32)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c2)
    # (64, 64, 32) -> (32, 32, 32)
    p2 = MaxPooling2D((2, 2)) (c2)
    # (32, 32, 32) -> (32, 32, 64)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p2)
    c3 = Dropout(0.2) (c3)
    # (32, 32, 64) -> (32, 32, 64)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c3)
    # (32, 32, 64) -> (16, 16, 64)
    p3 = MaxPooling2D((2, 2)) (c3)
    # (16, 16, 64) -> (16, 16, 128)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p3)
    c4 = Dropout(0.2) (c4)
    # (16, 16, 128) -> (16, 16, 128)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c4)
    # (16, 16, 128) -> (8, 8, 128)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    # (8, 8, 128) -> (8, 8, 256)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p4)
    c5 = Dropout(0.3) (c5)
    # (8, 8, 256) -> (8, 8, 256)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c5)
    
    # (8, 8, 256) -> (16, 16, 128)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c5)
    # (16, 16, 256)
    u6 = concatenate([u6, c4])
    # (16, 16, 256) -> (16, 16, 128)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u6)
    c6 = Dropout(0.2) (c6)
    # (16, 16, 128) -> (16, 16, 128)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c6)
    # (16, 16, 128) -> (32, 32, 64)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c6)
    # (32, 32, 128)
    u7 = concatenate([u7, c3])
    # (32, 32, 128) -> (32, 32, 64)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u7)
    c7 = Dropout(0.2) (c7)
    # (32, 32, 64) -> (32, 32, 64)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c7)
    # (32, 32, 64) -> (64, 64, 32)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c7)
    # (64, 64, 64)
    u8 = concatenate([u8, c2])
    # (64, 64, 64) -> (64, 64, 32)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u8)
    c8 = Dropout(0.1) (c8)
    # (64, 64, 32) -> (64, 64, 32)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c8)
    # (64, 64, 32) -> (128, 128, 16)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c8)
    # (128, 128, 16) -> (128, 128, 32)
    u9 = concatenate([u9, c1], axis=3)
    # (128, 128, 32) -> (128, 128, 16)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u9)
    c9 = Dropout(0.1) (c9)
    # (128, 128, 16) -> (128, 128, 16)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c9)
    # (128, 128, 16) -> (128, 128, 1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MeanIoU])
    model.summary()
    return model
    
    
def Train(model, X_train, Y_train):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
    if SAVE_LOAD:
        callbacks=[earlystopper, checkpointer]
    else:
        callbacks=[earlystopper]

    results = model.fit(X_train, Y_train, verbose=2, validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)
    return results
    
def CheckPrediction(model, X_train, Y_train):
    ix = random.randint(0, Y_train.shape[0])
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
    data = X_train[ix].reshape([1] + list(X_train[ix].shape))
    _, pred = Prediction(model, data)
    imshow(np.squeeze(pred))
    plt.show()


if __name__=='__main__':
    # X_train, Y_train, X_test, sizes_test = GetData()
    # TestData(X_train, Y_train, X_test)
    # model = GetUNetModel(reg)
    Train(model, X_train, Y_train)
    CheckPrediction(model, X_train, Y_train)
    df = GetSubmissionCSV(model, X_test, sizes_test)

    
    
    
    
    

