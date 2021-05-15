#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""


#possible computations; pairwise distances
#multiple types of amplitudes; entropy

from gtda.diagrams import Amplitude
from gtda.homology import CubicalPersistence
from gtda.images import HeightFiltration,Binarizer
import cv2

import numpy as np
import time,fnmatch,os
from persim import PersImage

# =============================================================================
# Variables to set
#
# objSegMapDir is the path to the directory that stores object segmentation maps
# for which persistence features are to be extracted
# 
# augmentation is a boolean variable; when set to true it generates persistence images for 
# rotated object segmentation maps as well from the corresponding persistence diagrams.
# Note that for this to work, augmentation should be set true when generating
# persistence diagrams as well.
# 
# makedirs is a boolean variable; when set to true it creates directories 
# to store the generated amplitude features
# =============================================================================
    
# %%

objSegMapDir = './uwis/livingRoom/objMasks/'
image_files = fnmatch.filter(os.listdir(objSegMapDir), '*.png') 


augmentation = True #set true if generating PIs for object segmentation maps of scenes training environment
makedirs = True
if makedirs:
    
    os.mkdir(objSegMapDir+'pis')
    for direction in range(16):
        if direction%2==0:
            
            os.mkdir(objSegMapDir+'pis/'+str(direction))


for ctri in range(len(image_files)):
    filename = image_files[ctri]

    if augmentation:
        numdiag = 4
    else:
        numdiag = 1

    for i in range(16):
        if i%2==0:
            for rot in range(numdiag):
                diag = np.load(objSegMapDir+'pds/'+str(i)+'/'+os.path.splitext(filename)[0]+'_r'+str(rot)+'_d'+str(i)+'.npy')
                pim = PersImage(spread=20, pixels=[50,50], verbose=False)
                img = pim.transform(diag)
                np.save(objSegMapDir+'pis/'+str(i)+'/'+os.path.splitext(filename)[0]+'_r'+str(rot)+'_d'+str(i)+'.npy',img)

            
