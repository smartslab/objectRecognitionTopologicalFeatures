#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""
import fnmatch,os
from sklearn.model_selection import train_test_split
import numpy as np

import cv2

# %%
# =============================================================================
# Variables to set
# 
# sceneImagesDir is the path to the directory where scene images are stored
# 
# imageSetsDir is the path to the directory where training validation splits (if required) for deeplab are created
# 
# =============================================================================

# %%
sceneImagesDir = './JPEGImages_livingroom/'
imageSetsDir = './ImageSets_livingroom/'

image_files = fnmatch.filter(os.listdir(sceneImagesDir), '*.jpg')

list_no_ext = []
for i in range(len(image_files)):
    filename = image_files[i]
    name = os.path.splitext(filename)[0]
    list_no_ext.append(name)
    
x_train, x_test = train_test_split(list_no_ext,test_size=0.2, random_state=2018)


with open(imageSetsDir+'trainval.txt', 'w') as filehandle:
    for listitem in list_no_ext:
        filehandle.write('%s\n' % listitem)

with open(imageSetsDir+'train.txt', 'w') as filehandle:
    for listitem in x_train:
        filehandle.write('%s\n' % listitem) 

with open(imageSetsDir+'val.txt', 'w') as filehandle:
    for listitem in x_test:
        filehandle.write('%s\n' % listitem)
