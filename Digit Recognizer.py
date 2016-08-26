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
def svd(X):
    mat = []
    for i in range(len(X)):
        mat.append(X[i].reshape(28,28))
    return mat

mat = svd(X)
        
def single(x):
    u, sig, v = np.linalg.svd(x)
    sig16 = sig[:16]
    sigmat = np.zeros((16,16))
    for i in range(len(sig16)):
        sigmat[i][i] = sig16[i]
    newmat = np.dot(np.dot(u[:,:16], sigmat),v[:16,:])
    newrow = newmat.reshape(784)
    return newrow

def trainDataset(mat):
    newtrain = []
    for i in range(len(mat)):
        newrow = single(mat[i])
        newtrain.append(newrow)
    return newtrain
    
newtrain = trainDataset(mat)
        
a=single(mat[0])        
        
        
'''    
def sig(mat):
    sig16 = []
    U = []
    V = []
    for i in range(len(mat)):
        u,sig,v = mat[i]
        sigma = sig[:16]
        sig16.append(sigma)
        U.append(u)
        V.append(v)
    return sig16, U, V
'''    
   

    
    
    
        
        
    
        
        
    
def trainNB(mat, category):
    m, n= np.shape(mat)
    pClass1 = sum(category) + 1 / float(m + n)
    p0Num = np.zeros(n)
    p1Num = np.zeros(n)
    p2Num = np.zeros(n)
    p3Num = np.zeros(n)
    p4Num = np.zeros(n)
    p5Num = np.zeros(n)
    p6Num = np.zeros(n)
    p7Num = np.zeros(n)
    p8Num = np.zeros(n)
    p9Num = np.zeros(n)
    p0Denom = 0.0; p1Denom = 0.0
    p2Denom = 0.0; p3Denom = 0.0
    p4Denom = 0.0; p5Denom = 0.0
    p6Denom = 0.0; p7Denom = 0.0
    p8Denom = 0.0; p9Denom = 0.0
    for i in range(m):
        if category[i] == 0:
            p0Num += mat[i]
            p0Denom += 1
        elif category[i] == 1:
            p1Num += mat[i]
            p1Denom += 1
        elif category[i] == 2:
            p2Num += mat[i]
            p2Denom += 1
        elif category[i] == 3:
            p3Num += mat[i]
            p3Denom += 1
        elif category[i] == 4:
            p4Num += mat[i]
            p4Denom += 1
        elif category[i] == 5:
            p5Num += mat[i]
            p5Denom += 1
        elif category[i] == 6:
            p6Num += mat[i]
            p6Denom += 1
        elif category[i] == 7:
            p7Num += mat[i]
            p7Denom += 1
        elif category[i] == 8:
            p8Num += mat[i]
            p8Denom += 1
        else:
            p9Num += mat[i]
            p9Denom += 1
            
    p1vec = np.array(p1Num) + 1 / p1Denom + 10
    p0vec = np.array(p0Num) + 1 / p0Denom + 10
    return p1vec, p0vec, pClass1
   
def classifyNB(vec2Classify, p1vec, p0vec, pClass1):
    m = np.shape(vec2Classify)[0]
    result =[]
    for i in range(m):
        p1 = np.sum(vec2Classify[i] * p1vec) + np.log(pClass1)
        p0 = np.sum(vec2Classify[i] * p0vec) + np.log(1-pClass1)
        if p1 > p0:
            result.append(1)
        else:
            result.append(0)
    return result
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
    
'''
#loss function
theta = np.ones((n+1, 1))
def loss(x, y, theta, lamda):
    z = x.dot(theta)
    reg = 0.5 * lamda * np.sum(np.array(theta)**2)
    L = y.T.dot(np.log(sigmoid(z)))+(1-y).T.dot(np.log(1 - sigmoid(z))) + reg
    return L
    
def sga(x, y, lamda, num_iters=500):
    m, n = np.shape(x)
    theta = np.ones(n)
    for i in range(num_iters):
        for j in range(n):
        #for j in xrange(m):
            alpha = 0.01 + 4 / (1+i+j)
            randindex = int(np.random.uniform(0, m))
            h = sigmoid(np.sum(x[randindex] * theta))
            if j == 0:
                theta[j] = theta[j] + alpha * ((y[randindex] - h) * x[randindex,0])
            else:
                theta[j] = theta[j] + alpha * ((y[randindex] - h) * x[randindex,j] + lamda * theta[j])
    return theta
'''

def cost(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg
    
def gradient(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    #parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel() 
    
from scipy.optimize import minimize   
 
def one_vs_all(X, y, num_labels, learning_rate):  
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params + 1))
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta
    
all_theta = one_vs_all(X, y, 10, 0.1)
    
def predict_all(X, all_theta):  
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    h = sigmoid(X * all_theta.T)

    h_argmax = np.argmax(h, axis=1)

    return h_argmax
    
y_pred = predict_all(test, all_theta)  
presult=pd.DataFrame(y_pred, columns=['Label'])
ImageId = np.arange(1,np.shape(test)[0]+1)
Id = pd.DataFrame(ImageId, columns=['ImageId'])
presult['ImageId'] = Id['ImageId']





