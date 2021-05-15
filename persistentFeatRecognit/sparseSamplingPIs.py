# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""


import numpy as np
import os,fnmatch

from sklearn.model_selection import train_test_split
from optimal_sparse_sampling.optimalMeasurements import optimalMeasurements
import time

# =============================================================================
# Variables to set
#
# objSegMapDir is the path to the directory that stores object semgentation maps
# for which persistence features are to be extracted
# 
# foldnum: Choose one of the five training test splits of scene images from the environment
# on which models are to be trained
# 
# makedirs is a boolean variable; when set to true it creates directories 
# to store the optimal pixel locations
# =============================================================================



# %%
# Divide dataset into five folds.

# Generate five training and test folds of scene images from the environment
# on which models are to be trained

# This is training and test fold splitting scheme is what we used for the UW-IS dataset.
# Alternatively use K-Fold generator from scikit-learn 

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

# %%
objSegMapDir = './uwis/livingRoom/objMasks/' #path to directory with object segmentation maps
foldnum = 1 #choose from 1,2,3,4,5

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



train_pi_names, val_pi_names, train_labels, val_labels = train_test_split(pi_files, label_files, test_size=0.2, stratify=label_files, random_state=2018)


makedirs = True

if makedirs:
    dirname ='pi_locs_train_fold'+str(foldnum)
    os.mkrdir(objSegMapDir+'pi_locs/')
    os.mkdir(objSegMapDir+'pi_locs/'+dirname)
    os.mkdir(objSegMapDir+'pi_locs/'+dirname+'/8dirs/')


for dirnum in range(16):
    if dirnum%2==0:
        train_pis = np.zeros((len(train_pi_names),2500))
        for i in range(len(train_pi_names)):
            filename = train_pi_names[i]
            img = np.load(objSegMapDir+'pis/' + str(dirnum)+'/'+os.path.splitext(filename)[0]+'_d'+str(dirnum)+'.npy')
            if len(np.argwhere(np.isnan(img))) > 0:
                donothing = 0
                print(filename)
            else:
                train_pis[i,:] = np.reshape(img,(2500,))
        print(len(train_pi_names))
        print(np.shape(train_pis))
        print('[INFO] PI sparse sampling begins for direction  ',dirnum )
        t1=time.time()
        pi_r_opt,_,pi_loc = optimalMeasurements(np.transpose(train_pis),2000)
        t2=time.time()
        print('[INFO] QR sampling complete for PIs .. r_opt is',pi_r_opt)
        print('[INFO] Time required', t2-t1)
        

        with open(objSegMapDir+'pi_locs/'+dirname+'/pi_locs_train_dir'+str(dirnum)+'opt_'+str(pi_r_opt)+'.npy', 'wb') as f3:
            np.save(f3, pi_loc)
        with open(objSegMapDir+'pi_locs/'+dirname+'/8dirs/pi_locs_train_dir'+str(dirnum)+'.npy', 'wb') as f4:
            np.save(f4, pi_loc)
