#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 21:55:41 2021

@author: smartslab
"""

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2,fnmatch
from sklearn.model_selection import train_test_split

import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau



# %%
# =============================================================================
# Variables to set
#
# objSegMapDir is the path to the directory that stores object segmentation maps
# obtained from scene images of the unseen environment on which models have to be tested
# 
# foldnum: Choose one of the five training test splits of scene images from the environment
# on which models are to be trained
# 
# filepath is the filepath for the trained model
# =============================================================================


# %%


def classifier_mlp_softmax(n_classes=14):
    classifier = Sequential()

    classifier.add(Dense(512, input_shape = (8,)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(256))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))
    
    
    classifier.add(Dense(n_classes))
    classifier.add(BatchNormalization())
    classifier.add(Activation('softmax'))
    
    return classifier


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 50, 100, 150, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-2
    if epoch > 250:
        lr *= 1e-2
    elif epoch > 200:
        lr *= 1e-2
    elif epoch > 100:
        lr *= 1e-1
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

objSegMapDir = './uwis/warehouse/objMasks/' #path to directory with object segmentation maps
test_crops = fnmatch.filter(os.listdir(objSegMapDir), '*.png')

foldnum = 1

augmentation = False
pi_files = []

for i in range(len(test_crops)):
    filename = test_crops[i]
    name = os.path.splitext(filename)[0]
    if augmentation:
        numdiag = 4       
    else:
        numdiag = 1
    for rot in range(numdiag):
            pi_files.append(name +'_r'+str(rot)+'.npy')
        
classes = ['obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14']
        
label_files = []
ctrl = 0
for i in range(len(pi_files)):
    filename = pi_files[i]
    name = os.path.splitext(filename)[0]
    splitname = name.split('_')
    classname = splitname[-2]
    label = classes.index(classname)
    label_files.append(label)
    ctrl = ctrl+1


features = np.zeros((len(pi_files),8))

for i in range(len(pi_files)):
    filename = pi_files[i]
    ampl = np.load(objSegMapDir+'ampls/'+filename)
    features[i,:] = ampl
    

x_test = features

y_test = np.asarray(label_files)

model = classifier_mlp_softmax(len(classes))

filepath = './amplitude_livingroom_fold'+str(foldnum)+'.hdf5'
model.load_weights(filepath)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr_schedule(0)),metrics=['accuracy'])

y_pred = model.predict_classes(x_test)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names = classes)
print(report)