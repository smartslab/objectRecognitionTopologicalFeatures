#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""



from gtda.diagrams import PersistenceEntropy,Amplitude
from gtda.homology import VietorisRipsPersistence,CubicalPersistence
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
# augmentation is a boolean variable; when set to true it generates amplitudes for 
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
    os.mkdir(objSegMapDir+'ampls')
            

for ctri in range(len(image_files)):
    filename = image_files[ctri]

    if augmentation:
        numdiag = 4
    else:
        numdiag = 1
    for rot in range(numdiag):
            
        for i in range(16):
            if i%2==0:
                diag = np.load(objSegMapdir+'pds/'+str(i)+'/'+os.path.splitext(filename)[0]+'_r'+str(rot)+'_d'+str(i)+'.npy')
                homog = np.zeros((np.shape(diag)[0],1))
                dgm = np.concatenate((diag,homog),axis=1)
                amptr = Amplitude(metric = 'bottleneck')
                curr_ampl = amptr.fit_transform(np.asarray([dgm]))
                if i==0:
                    ampl = curr_ampl
                else:
                    ampl = np.concatenate((ampl, curr_ampl),axis=1)
        
        np.save(objSegMapDir+'ampls/'+os.path.splitext(filename)[0]+'_r'+str(rot)+'.npy',ampl)
