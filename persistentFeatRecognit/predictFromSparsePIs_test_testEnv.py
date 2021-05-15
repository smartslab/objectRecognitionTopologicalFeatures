#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:02:21 2021

@author: smartslab
"""

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import cv2,fnmatch
from sklearn.model_selection import train_test_split

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


def classifier_mlp_softmax_3(n_classes=70):
    classifier = Sequential()
    classifier.add(Dense(2048, input_shape = (3433,)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(1024))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(512))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(256))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    
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
    if epoch > 1000:
        lr *= 1e-2
    elif epoch > 700:
        lr *= 1e-2
    elif epoch > 600:
        lr *= 1e-1
    elif epoch > 500:
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

foldername ='pi_locs_train_fold'+str(foldnum)

#optimal pixel locations obtained using sparse sampling
all_r_opts = {}
all_r_opts[1] = [373,0,450,0,381,0,452,0,387,0,457,0,386,0,457,0]#fold1
all_r_opts[2] = [373,0,451,0,386,0,458,0,388,0,457,0,390,0,460,0]#fold2
all_r_opts[3] = [376,0,453,0,384,0,453,0,387,0,455,0,386,0,457,0]#fold3
all_r_opts[4] = [374,0,460,0,386,0,462,0,387,0,463,0,389,0,460,0]#fold4
all_r_opts[5] = [373,0,451,0,384,0,454,0,386,0,454,0,387,0,460,0]#fold5


for dirnum in range(16):
    if dirnum%2==0:
        pis = np.zeros((len(pi_files),2500))
        for i in range(len(pi_files)):
            filename = pi_files[i]
            img = np.load(objSegMapDir+'pis/' + str(dirnum)+'/'+os.path.splitext(filename)[0]+'_d'+str(dirnum)+'.npy')
            pis[i,:] = np.reshape(img,(2500,))
        
        pi_loc = np.load(objSegMapDir+'pi_locs/'+foldername+'/8dirs/pi_locs_train_dir'+str(dirnum)+'.npy')
        pi_samples = pi_loc[0:r_opts[dirnum]]
        curr_features = pis[:,pi_samples]
        if dirnum==0:
            features = curr_features
        else:
            features = np.concatenate((features, curr_features),axis=1)


x_test = features

y_test = np.asarray(label_files)



model = classifier_mlp_softmax_3(len(classes))


filepath = './sparse_pi_livingroom_fold'+str(foldnum)+'.hdf5'

model.load_weights(filepath)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr_schedule(0)),metrics=['accuracy'])

y_pred = model.predict_classes(x_test)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names = classes)
print(report)