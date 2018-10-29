# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:21:49 2018

@author: silus
"""
import numpy as np
import sys
sys.path.append('../')
from methods.implementations import svm_classification, predict_svm_outcome, reg_logistic_regression, logistic_prediction

def normalize_data(data):
    
    data[data == -999] = np.nan
    
    features_mean = np.nanmean(data, axis=0)
    features_stdev = np.nanstd(data, axis=0)

    tmp = (data-features_mean)/features_stdev
    #tmp[np.isnan(tmp)] = -999

    return tmp, features_mean, features_stdev

def cross_validation(y, tx, k, lambdas, method, n_iter, gamma):
    "Cross validation for method specified as a str in the method argument"
    method_mapping = {'logistic': (reg_logistic_regression,logistic_prediction),
                      'svm': (svm_classification, predict_svm_outcome)}
    
    n_data = len(y)
    ind = np.random.permutation(n_data)
    ind = ind[0:k*int(np.floor(n_data/k))] #Remove additional data so that each batch has the same size
    ind = ind.reshape(k,-1)
 
    #Optimize lambda
    best_acc = 0
    initial_w = np.zeros([tx.shape[1]])
    for lambda_ in lambdas:
        acc_tot = 0 #reinitialize to 0
        loss_tot = 0
        for i in range(k):
            #Divide data to train and val
            ind_tr = np.append(ind[:i,:],ind[i+1:,:])
            tx_tr = tx[ind_tr,:]
            y_tr = y[ind_tr]
            tx_val = tx[ind[i,:],:]
            y_val = y[ind[i]]
            #Train
            w, loss = method_mapping[method][0](y_tr, tx_tr, lambda_, initial_w, n_iter, gamma)
            loss_tot += loss
            #Test on validation set
            pred = np.round(method_mapping[method][1](tx_val,w))
            acc_tot += np.mean(y_val==pred)
        #Calculate average over all validation batches
        acc = acc_tot/k
        loss_ = loss_tot/k
        print("Method: " + method +" Lambda: %3.1e, Loss: %3.3e, Accuracy: %3.3f%%" % (lambda_, loss_, acc*100))
        #Update best values if current is better
        if acc > best_acc:
            best_acc = acc
            best_lambda = lambda_
    print('')
    print("Best lambda: %3.3e, Accuracy: %3.3f%%" % (best_lambda, best_acc*100))
    return best_acc, best_lambda


        
    
    