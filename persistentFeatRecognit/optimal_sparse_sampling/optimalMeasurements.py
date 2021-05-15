# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:28 2020

@author: Ekta Samani
"""
#code for optimalMeasurements.m

#IN:
#   X: Input m*n data matrix
#   p: number of desired measurements

# OUT: 
#    r_opt: Gavish-Donoho optimal rank truncation
#    energy: energy contained in first r_opt modes
#    loc: optimal measurement indices of m-dim states

import numpy as np
import statistics
import scipy
from optimal_sparse_sampling.optimal_SVHT_coef import *
def optimalMeasurements(X,p):
    U,S,_ = np.linalg.svd(X,full_matrices=False)
    
    #update to use Gavish-Donoho truncation paramter
    m = min(np.shape(X))
    n = max(np.shape(X))
    #diagvec = np.diagonal(S).copy()
    diagvec = S.copy()
    sig = diagvec/np.sum(diagvec)
    
    betavec = np.asarray([m/n])
    thres = optimal_SVHT_coef(betavec,0)*statistics.median(sig)
    #thres = optimal_SVHT_coef(m/n,0)*statistics.median(sig)
    r_opt = np.prod(np.shape(sig[np.where(sig > thres)]))
    
    print('Gavish--Donoho optimal rank truncation=', r_opt)
    
    energy = np.cumsum(diagvec)/np.sum(diagvec)
    print('energy contained in first r_opt modes=', energy[r_opt -1]*100, '%');
    #remember python indices start with 0!
    
    Ur = U[:,0:r_opt]
    Urt = np.transpose(Ur)
    
    assert p>=r_opt
    
    if (p==r_opt):
        _,_,pivot = scipy.linalg.qr(Urt,mode='economic',pivoting=True)
    else:
        _,_,pivot = scipy.linalg.qr(Ur.dot(Urt),mode='economic',pivoting=True)
        
    loc = pivot[0:p]
    
    return r_opt,energy,loc

# pis = np.load('optimal_sparse_sampling/pi_features_train.npy')
# #X = np.random.random((100,30))
# p = 5000
# r_opt,_,loc = optimalMeasurements(pis[1:10000,:],p) 

