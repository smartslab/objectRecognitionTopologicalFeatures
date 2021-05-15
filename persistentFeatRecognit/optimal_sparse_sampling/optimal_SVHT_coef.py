# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:38:54 2020

@author: Ekta Samani
"""

import numpy as np

from scipy import integrate

def optimal_SVHT_coef(beta,sigma_known):
    if sigma_known:
        coef=optimal_SVHT_coef_sigma_known(beta)
    else:
        coef=optimal_SVHT_coef_sigma_unknown(beta)
    return coef

def optimal_SVHT_coef_sigma_known(beta):
    assert np.all(beta>0)
    assert np.all(beta<=1)
    #assert beta.ndim == 1 #beta must be a vector
    
    w = (8*beta)/(beta+1+np.sqrt(np.power(beta,2) + 14*beta + 1))
    lambda_star = np.sqrt(2*(beta + 1) + w)
    return lambda_star

def optimal_SVHT_coef_sigma_unknown(beta):
    #warning off
    assert np.all(beta>0)
    assert np.all(beta<=1)
    #assert beta.ndim == 1 #beta must be a vector
    
    coef = optimal_SVHT_coef_sigma_known(beta)
    
    MPmedian = np.zeros(np.shape(beta))
    
    for i in range(beta.size):
        MPmedian[i] = MedianMarcenkoPastur(beta[i])
    
    omega = coef/np.sqrt(MPmedian)
    
    return omega

def MarcenkoPasturIntegral(x,beta):
    #changed this to an assert instead of 
    #error message
    assert np.all(beta>0)
    assert np.all(beta<=1)
    
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    
    assert x >= lobnd
    assert x <= hibnd
    
    dens = lambda t: np.sqrt(np.multiply((hibnd-t),(t-lobnd)))/(np.multiply(2*np.pi*beta,t))
    I = integrate.quad(dens,lobnd,x) 
    print('x=', x, 'beta=', beta, 'I=', I[0])
    return I[0]
    

def MedianMarcenkoPastur(beta):
    MarPas = lambda x: 1-incMarPas(x,beta,0)
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    
    change = 1
    while (change and (hibnd - lobnd > 0.001)):
        change = 0
        x = np.linspace(lobnd,hibnd,5)
        y = np.zeros_like(x)
        xlen = beta.size
        for i in range(xlen):
            y[i]=MarPas(x[i])
        if np.any(y<0.5):
            lobnd = max(x[y<0.5])
            change=1
        if np.any(y>0.5):
            hibnd = min(x[y>0.5])
            change=1
    med = (hibnd+lobnd)/2
    return med

def IfElse(Q,point,counterPoint):
    y = point
    if np.any(Q==0):
        if counterPoint.size == 1:
            counterPoint = np.ones(np.shape(Q))*counterPoint
        y[np.where(Q==0)] = counterPoint[np.where(Q==0)]
    return y
           
def incMarPas(x0,beta,gamma):
    #assert(beta<=1)
    topSpec = (1 + np.sqrt(beta))**2
    botSpec = (1 - np.sqrt(beta))**2
    MarPas = lambda x: IfElse((topSpec - x)*(x-botSpec) > 0,np.sqrt((topSpec-x)*(x-botSpec))/(beta*x)/(2*np.pi),0) 
    if gamma != 0:
        fun = lambda x:(x**gamma)*MarPas(x)
    else:
        fun = lambda x:MarPas(x)
    I = integrate.quad(fun,x0,topSpec)        
    return I[0]