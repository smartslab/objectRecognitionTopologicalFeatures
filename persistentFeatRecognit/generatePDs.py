#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""



from gtda.diagrams import Amplitude
from gtda.homology import CubicalPersistence
from gtda.images import HeightFiltration,Binarizer
import cv2

import numpy as np
import time,fnmatch,os

# =============================================================================
# Variables to set
#
# objSegMapDir is the path to the directory that stores object segmentation maps
# for which persistence features are to be extracted
# 
# augmentation is a boolean variable; when set to true it generates persistence diagrams for 
# rotated object segmentation maps as well. Set it to true when generating persistence diagrams
# for object segmentation maps from the training environment
# 
# makedirs is a boolean variable; when set to true it creates directories 
# to store the generated persistence diagrams
# =============================================================================

    
# %%

objSegMapDir = './uwis/livingRoom/objMasks/'
image_files = fnmatch.filter(os.listdir(objSegMapDir), '*.png') 


augmentation = True #set true if generating PIs for object segmentation maps of scenes training environment
makedirs = True
if makedirs:
    os.mkdir(objSegMapDir+'pds')

    for direction in range(16):
        if direction%2==0:
            os.mkdir(objSegMapDir+'pds/'+str(direction))

for ctri in range(len(image_files)):
    filename = image_files[ctri]

    
    orig_mask = cv2.imread(objSegMapDir+filename,0)
    #padding and resizing
    w = np.shape(orig_mask)[0]
    h = np.shape(orig_mask)[1]
    mask = np.full((513,513),(0),dtype=np.uint8)
    xx = (513-w)//2
    yy = (513-h)//2
    mask[xx:xx+w,yy:yy+h] = orig_mask
    

    ret,thresh1 = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
    
    print('Image read: ' + str(ctri) + 'size'+ str(np.shape(thresh1)[0]) + str(np.shape(thresh1)[1]))

    grayscale = cv2.resize(thresh1, (125,125), interpolation = cv2.INTER_AREA)
    
    inputs = []
    inputs.append(grayscale)
    if augmentation:
        for i in range(1,4):
            inputs.append(np.rot90(grayscale,i))
        
            
    arrayOfInputs = np.asarray(inputs)
    br = Binarizer()
    
    binarized= br.fit_transform(arrayOfInputs)
    for i in range(16):
        if i%2==0:
            hf = HeightFiltration(direction=np.array([np.cos(i*(np.pi/8)),np.sin(i*(np.pi/8))])) #filtration using height functions
            inputdata = hf.fit_transform(binarized)
            showimg = inputdata[0,:,:]
            #apply persistent homology
            cubical_tr = CubicalPersistence(homology_dimensions=[0])
            diagrams = cubical_tr.fit_transform(inputdata) 
    
            numdiag = np.shape(diagrams)[0]
            for rot in range(numdiag):
                np.save(objSegMapDir+'pds/'+str(i)+'/'+os.path.splitext(filename)[0]+'_r'+str(rot)+'_d'+str(i)+'.npy',diagrams[rot][:,0:2])
                

    
