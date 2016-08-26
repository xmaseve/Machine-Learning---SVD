# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:45:05 2016

@author: YI
"""

import numpy as np
import pandas as pd


train = pd.read_csv('C:/Users/YI/Downloads/Dtrain.csv')
test = pd.read_csv('C:/Users/YI/Downloads/Dtest.csv')
    
label = train['label']
train.drop('label', axis=1,inplace=True)

X = train.as_matrix()
y = label.as_matrix()
test = test.as_matrix()

#make every row in X convert to 28*28 matrix for SVD
def convert_matrix(X):
    mat = []
    for i in range(len(X)):
        mat.append(X[i].reshape(28,28))
    return mat

        
def single(x):
    u, sig, v = np.linalg.svd(x)
    sig16 = sig[:16]
    sigmat = np.zeros((16,16))
    for i in range(len(sig16)):
        sigmat[i][i] = sig16[i]
    newmat = np.dot(np.dot(u[:,:16], sigmat),v[:16,:])
    newrow = newmat.reshape(784)
    return newrow

#reconstitue the matrix
def trainDataset(mat):
    newtrain = []
    for i in range(len(mat)):
        newrow = single(mat[i])
        newtrain.append(newrow)
    return newtrain
    
       
        
        

   

    
    
    
        
        
    
        
        
    

        
