import os
import sys
import random

import numpy as np
import pandas as pd
from skimage import measure

import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imshow
from skimage.transform import resize


from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from numpy import logical_and as AND
from numpy import logical_not as NOT

import tensorflow as tf
from data import GetData
from model import GetUNetModel
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
    

    
    
def Train(model, X_train, Y_train):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
    if SAVE_LOAD:
        callbacks=[earlystopper, checkpointer]
    else:
        callbacks=[earlystopper]

    results = model.fit(X_train, Y_train, verbose=2, validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)
    return results
    
# Data has shape of [batch, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
def Prediction(model,Data, threshold=0.5):
    if SAVE_LOAD:
        model = load_model('model-dsbowl2018-1.h5', custom_objects={'MeanIoU': MeanIoU})
    assert len(Data.shape)==4
    prob = model.predict(Data, batch_size=16, verbose=False)
    pred = (prob > threshold).astype(np.uint8)
    return prob, pred

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

'''
Encode Prediction to String
input: a binary ndarray image of shape (W, H)
    for example:
    0 1 1
    1 0 0
    1 1 0
output: a string of indexes
    for the previous example:
    '2 3 6 2'
    
for more details about the output format, plz visit
https://www.kaggle.com/c/data-science-bowl-2018#evaluation
'''
def PredictionToString(pred, is_test = False):
    ArrayToString = lambda x: ' '.join(str(y) for y in x)
    content = pred.T.ravel()
    index = np.array(range(len(content)))+1
    content_pre = np.concatenate([[0], content[:-1]])
    content_post = np.concatenate([content[1:], [0]])
    index_pre = index[AND(content, NOT(content_pre))]
    index_post = index[AND(content, NOT(content_post))]
    length = index_post - index_pre + 1
    if is_test:
        assert len(index_pre)==len(index_post)
        for l in length:
            assert l >= 1
    picked = []
    for i in range(len(index_pre)):
        picked.append(index_pre[i])
        picked.append(length[i])
    return ArrayToString(picked)
    
    
'''
Encode A labeled image to a list of encoded strings
    input: a image of shape (W, H), 
    each nucleus is assigned with one unique label, for example:
    0 2 2 0 3
    1 0 2 0 3
    1 0 0 0 3
    represents a image with 3  nucleus. label 0 represents the background
    
    output: a list of encodings. each element in the list represents a encoded nucleus

'''
def GetEncoded(labeled_img):
    num_nucleus = np.max(labeled_img)
    encodes = []
    for label in range(num_nucleus):
        if label==0:
            continue
        current_nucles = labeled_img==label
        encodes.append(PredictionToString(current_nucles))
    return encodes
        
def GetSubmissionCSV(model, X_test, sizes_test):
    probabilities, _ = Prediction(model, X_test, threshold=0.5)
    list_path = next(os.walk(TEST_PATH))[1]
    
    results = []
    for i in tqdm(range(len(probabilities))):
        resized_prob = (resize(np.squeeze(probabilities[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))
        pred = resized_prob > 0.5
        labeled_pred = measure.label(pred, neighbors=4 ,background=0)
        encodes = GetEncoded(labeled_pred)
        results += [[list_path[i], encode] for encode in encodes]
        
    sub = pd.DataFrame(results,columns=['ImageId','EncodedPixels'])
    sub.to_csv('sub.csv', index=False)
    return sub


if __name__=='__main__':
    X_train, Y_train, X_test, sizes_test = GetData()
    # TestData(X_train, Y_train, X_test)
    model = GetUNetModel(reg)
    Train(model, X_train, Y_train)
    CheckPrediction(model, X_train, Y_train)
    df = GetSubmissionCSV(model, X_test, sizes_test)

    
    
    
    
    

