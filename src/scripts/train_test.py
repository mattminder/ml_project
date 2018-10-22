# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:48:32 2018

@author: Zora
"""

#TODO: Load modified data (normalized and without missing values)
#TODO: Get best lambda from cross_validation
#TODO: Choose values (max_iter, gamma, ...)
#TODO: Get ids for submission

import numpy as np
import sys
sys.path.append('../')
from methods.implementations import reg_logistic_regression, logistic_prediction
from methods.proj1_helpers import create_csv_submission

#Load data as saved after missing_value_imputation
#tx_tr = load(...)
#y_tr = load(...) #As 0/1, not as -1/1
#tx_te = load(...)
#y_te = load(...)
#ids = load(...)
tx_tr = np.array([[1,2,3],[1,3,1],[1,0,0],[1,8,4],[1,0,3],[1,6,2]]) #Random values to test
y_tr = np.array([0,1,1,0,0,0])
tx_te = np.array([[1,3,2],[1,0,2]])
ids = np.array([0,1])
raise NotImplementedError

#Choose lambda according to best result of cross-validation
lambda_ = 1e-5
raise NotImplementedError

#Train with a decay of gamma
n_decay = 11
n_iter = 100
w = np.zeros(tx_tr.shape[1])
for gamma in np.logspace(0,-10,n_decay):
    w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, w, n_iter, gamma)

#Prediction for test data
y_pred = np.round(logistic_prediction(tx_te,w))
y_pred[np.where(y_pred == 0)] = -1

#Create submission
name = 'submission'
create_csv_submission(ids, y_pred, name)

