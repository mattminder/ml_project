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
from helpers.custom_helpers import cross_validation, plot_accuracy

# Load data as saved after missing_value_imputation
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")
y_logistic = y.copy()
y_logistic[np.where(y == -1)] = 0 #Want 0/1 data for logistic regression, not -1/1


# Augment data with all 1 vector
tx = np.c_[np.ones(tx.shape[0]), tx[:,0:30]]

### Cross validation for regularized logistic regression and SVM classifier
# K-fold cross validation
K = 10

# Initial hyperparameter range
lambdas = np.logspace(-10, 0, base = 10, num = 30)

n_iter = 100000
gamma = 0.001


train_acc_log = []
test_acc_log = []

acc_log, lambda_log, loss_log, acc_te_log, acc_tr_log = cross_validation(y_logistic, tx, K, lambdas, 'logistic', n_iter, gamma)
#acc_svm, lambda_svm, loss_svm, acc_te_svm, acc_tr_svm = cross_validation(y, tx, K, lambdas, 'svm', n_iter, gamma)

train_acc_log.append(acc_tr_log)
test_acc_log.append(acc_te_log)

tx = np.column_stack((tx, tx[:, 1:31]**2))
acc_log, lambda_log, loss_log, acc_te_log, acc_tr_log = cross_validation(y_logistic, tx, K, lambdas, 'logistic', n_iter, gamma)

train_acc_log.append(acc_tr_log)
test_acc_log.append(acc_te_log)

plot_accuracy(train_acc_log, test_acc_log, lambdas, 'logistic')



