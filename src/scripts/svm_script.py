
from methods.implementations import *
from helpers.custom_helpers import *
import numpy as np
# -------------------------------
# LOADING
# Load data, augment with 1, convert result to 0/1
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")
tx = np.c_[np.ones(tx.shape[0]), tx]

tx_toLog = np.log(tx[:, 0:29] + 20)
tx_toLog_norm, features_mean, features_stdev = normalize_data(tx_toLog)
tx = np.column_stack((tx, tx_toLog_norm))


# Wasi so probiert ha a features:
#tx = tx[:, 0:29]
# Mit de erste 29: 64.1%
# Mit aune ohni imputed:
#ix = [1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,29]
#tx = tx[:, ix]
# Max Acc = 61.616%


lambdaVec = np.logspace(2^-80, 2^0, base = 2, num = 81)
best_acc, best_lambda = cross_validation(y, tx, 10, lambdaVec, "svm")

