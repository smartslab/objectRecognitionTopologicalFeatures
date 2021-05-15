
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""


import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2,fnmatch
from sklearn.model_selection import train_test_split

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
# obtained from scene images of the environment on which models are to be trained
# 
# foldnum: Choose one of the five training test splits of scene images from the environment
# on which models are to be trained
# 
# filepath is the filepath for saving the trained model
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

# %%
foldnum = 1 #choose from 1,2,3,4,5
#generate test folds
scenenumlist = np.arange(1,348)

tfold1 = []
tfold2 = []
tfold3 = []
tfold4 = []
tfold5 = []
for i in range(1,348):
    if i%5==0:
        tfold1.append(i)
    elif i%5==1:
        tfold2.append(i)
    elif i%5==2:
        tfold3.append(i)
    elif i%5==3:
        tfold4.append(i)
    else:
        tfold5.append(i)

testfolds = {}
testfolds[1] = tfold1
testfolds[2] = tfold2
testfolds[3] = tfold3
testfolds[4] = tfold4
testfolds[5] = tfold5

objSegMapDir = './uwis/livingRoom/objMasks/' #path to directory with object segmentation maps
obj_seg_map_files = fnmatch.filter(os.listdir(objSegMapDir), '*.png')


train_crops = []
test_crops = []
for i in range(len(obj_seg_map_files)):
    filename = obj_seg_map_files[i]
    name = os.path.splitext(filename)[0]
    scenenum = int(name.split('_')[1][5:])
    if scenenum not in testfolds[foldnum]:
        train_crops.append(filename)
    else:
        test_crops.append(filename)

    
classes = ['obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14']



        
augmentation = True
pi_files = []

for i in range(len(train_crops)):
    filename = train_crops[i]
    name = os.path.splitext(filename)[0]
    if augmentation:
        numdiag = 4
        for rot in range(numdiag):
            pi_files.append(name +'_r'+str(rot)+'.npy')

        
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
    
    

    
indices = np.arange(len(pi_files))
x_train, x_val, y_train, y_val, idx_train,idx_val = train_test_split(features, label_files, indices,test_size=0.2, stratify=label_files, random_state=2018)



train_encoded_labels = to_categorical(y_train)
val_encoded_labels = to_categorical(y_val)

model = classifier_mlp_softmax(len(classes))
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr_schedule(0)),metrics=['accuracy'])

filepath = './amplitude_livingroom_fold'+str(foldnum)+'.hdf5'

if not os.path.exists(os.path.dirname(filepath)):
 	os.makedirs(os.path.dirname(filepath))
if os.path.isfile(filepath):
    os.remove(filepath)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]


history = model.fit(x_train, train_encoded_labels,
			batch_size=32,
		epochs=200,
			validation_data=(x_val, val_encoded_labels),
			verbose=2,
			shuffle=True,
            callbacks=callbacks)

(loss, accuracy) = model.evaluate(x_val, val_encoded_labels,batch_size=64,verbose=1)
print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))
model.save_weights(filepath, overwrite=True)


