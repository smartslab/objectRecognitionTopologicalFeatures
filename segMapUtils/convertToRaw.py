#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""


import numpy as np
from PIL import Image
import fnmatch,os
import cv2

# %%

# =============================================================================
# Variables to set
# 
# sceneSegMapDir is the directory to store hand generated segmentation masks
# 
# rawAnnsDir is the directory where raw annotations
# (pixel intensity indicates class label i.e. 1 for foreground and 0 for background)
# are to be stored
# 
# makedirs is boolean variable; set to true to create the rawAnnsDir
# =============================================================================

# %%
sceneSegMapDir = './foreground_livingroom/'
rawAnnsDir = './foregroundraw_livingroom/'

makedirs = True
if makedirs:
    os.mkdir(rawAnnsDir)

image_files = fnmatch.filter(os.listdir(sceneSegMapDir), '*.png')
    
for i in range(len(image_files)):
    filename=image_files[i]
    img = np.asarray(Image.open(sceneSegMapDir+filename).convert('L'))
    pil_mask = Image.fromarray((img/255).astype(np.uint8),'L') #to keep 1 as 1 not 255
    pil_mask.save(rawAnnsDir+filename)
    
    
