#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""


import cv2
import numpy as np
import fnmatch, os


# %%
# =============================================================================
# Variables to set
# 
# sceneImagesDir is the path to the directory where scene images are stored
# 
# stageOnePredsDir is the path to the directory where scene segmentation maps obtained from step 1 are stored
# 
# croppedObjectsDir is the path to the directory where cropped objects are to be stored
# 
# makedirs is a boolean variable; set to true to create the croppedObjectsDir
# 
# =============================================================================


# %%

sceneImagesDir = './JPEGImages_livingroom/'
stageOnePredsDir = './sceneSegMaps_livingroom/'
croppedObjectsDir = './livingroom_crops/'

image_files = fnmatch.filter(os.listdir(sceneImagesDir), '*.jpg')
seg_files = fnmatch.filter(os.listdir(stageOnePredsDir), '*.png')

makedirs = True
if makedirs:
    os.mkdir(croppedObjectsDir)

#for i in range(1):

for i in range(len(image_files)):
    filename = image_files[i]
    name = os.path.splitext(filename)[0]
    stage1pred = cv2.imread(stageOnePredsDir+name+'.png')
    orig = cv2.imread(sceneImagesDir+name+'.jpg')
    if np.shape(orig)[1] > np.shape(orig)[0]:
        ratio = np.shape(orig)[1]/513
    else:
        ratio = np.shape(orig)[0]/513
    
    image = cv2.cvtColor(stage1pred, cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours,_ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    xs = []
    ys = []
    
    pad_width = 5
    
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        
        if (int(ratio*(y-pad_width)) < 0) or (int(ratio*(y+h +pad_width)) > np.shape(orig)[0]) or (int(ratio*(x-pad_width)) < 0) or (int(ratio*(x+w +pad_width)) > np.shape(orig)[1]):
            pad_width = 0

        cropped_img = orig[int(ratio*(y-pad_width)):int(ratio*(y+h +pad_width)), int(ratio*(x-pad_width)):int(ratio*(x+w+pad_width))] 
        cv2.imwrite(croppedObjectsDir+name+'_'+str(i)+'_cropped.jpg',cropped_img)
	



