# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:54:30 2018

@author: Zora
"""

#TODO: Load modified data (normalized and without missing values)
#TODO: Choose values (n_batches, lambdas, max_iter, gamma, ...)
#Maybe train gamma as well
#Rerun with lambdas close to the best value found in first try

import numpy as np
import sys
sys.path.append('../')
from methods.implementations import reg_logistic_regression, logistic_prediction

#Load data as saved after missing_value_imputation
#tx = load(...)
#y = load(...) #As 0/1, not as -1/1
tx = np.array([[1,2,3],[1,3,1],[1,0,0],[1,8,4],[1,0,3],[1,6,2]]) #To test the methods
y = np.array([0,1,1,0,0,0])
raise NotImplementedError

#Divide data indices for cross-validation
n_batches = 5
n_data = len(y)
ind = np.random.permutation(n_data)
ind = ind[0:n_batches*int(np.floor(n_data/n_batches))] #Remove additional data so that each batch has the same size
ind = ind.reshape(n_batches,-1)

#Optimize lambda
best_acc = 0
initial_w = np.zeros([tx.shape[1]])
for lambda_ in np.logspace(-5,5,11):
    acc_tot = 0 #reinitialize to 0
    for i in range(n_batches):
        #Divide data to train and val
        ind_tr = np.append(ind[:i,:],ind[i+1:,:])
        tx_tr = tx[ind_tr,:]
        y_tr = y[ind_tr]
        tx_val = tx[ind[i,:],:]
        y_val = y[ind[i]]
        #Train
        w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, 1000, 1e-4)
        #Test on validation set
        pred = np.round(logistic_prediction(tx_val, w))
        accuracy = np.mean(y_val==pred)
        acc_tot += accuracy
    #Calculate average over all validation batches
    acc = acc_tot/n_batches
    #Update best values if current is better
    if acc > best_acc:
        best_acc = acc
        best_lambda = lambda_
print('The best lambda is: ')
print(best_lambda)
print('Validation accuracy:')
print(best_acc)
