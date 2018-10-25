# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:54:30 2018

@author: Zora
"""

# Maybe train gamma as well
# Rerun with lambdas close to the best value found in first try

import numpy as np
import sys
sys.path.append('../')
from helpers.custom_helpers import cross_validation

# Load data as saved after missing_value_imputation
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")
y_logistic = y.copy()
y_logistic[np.where(y == -1)] = 0 #Want 0/1 data for logistic regression, not -1/1


# Augment data with all 1 vector
tx = np.c_[np.ones(tx.shape[0]), tx]

### Cross validation for regularized logistic regression and SVM classifier

# K-fold cross validation
K = 5

# Initial hyperparameter range
lambdas = np.logspace(-4,-2,11)

n_iter = 100000
gamma = 0.01


best_param_log = []
best_param_svm = []

acc_log, lambda_log = cross_validation(y_logistic, tx, K, lambdas, 'logistic', n_iter, gamma)
acc_svm, lambda_svm = cross_validation(y, tx, K, lambdas, 'svm', n_iter, gamma)

best_param_log.append((acc_log, lambda_log))
best_param_svm.append((acc_svm, lambda_svm))

