# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:18:31 2018

@author: Silvan
"""

import numpy as np
import sys
sys.path.append('../')
from helpers.custom_helpers import cross_validation

#Load data as saved after missing_value_imputation
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")

# K-fold cross validation
K = 5

# Initial hyperparameter range
lambdas = np.logspace(-4,-3,11)

best_acc = []
best_lambda = []

acc, lambda_ = cross_validation(y, tx, K, lambdas, 'svm')

best_acc.append(acc)
best_lambda.append(lambda_)

# Do a second cross validation for hyperparameter range around the best one 
refined_lambdas = np.linspace(lambda_-0.0001/2,lambda_+0.0001/2,11)

acc, lambda_ = cross_validation(y, tx, K, lambdas, 'svm')

best_acc.append(acc)
best_lambda.append(lambda_)
