
from methods.implementations import *
from helpers.custom_helpers import *
import numpy as np
from helpers.proj1_helpers import *
# -------------------------------
# LOADING
# Load data, expand to polynomials
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")


def x_expand(x):
    x1 = x[:,:-6] #without dummy variables
    tx = np.empty([x1.shape[0],int(x1.shape[1]*(x1.shape[1]-1)/2)])
    k = 0
    for i in range(x1.shape[1]):
        for j in range(i):
            tx[:,k] = x1[:,i]*x1[:,j]
            k += 1
    normalized = normalize_data(tx)
    return np.c_[x, normalized[0]]


tx = x_expand(tx)
tx = np.c_[np.ones(tx.shape[0]), tx]

#tx_toLog = np.log(tx[:, 0:29] + 20)
#tx_toLog_norm, features_mean, features_stdev = normalize_data(tx_toLog)
#tx = np.column_stack((tx, tx_toLog_norm))


# Wasi so probiert ha a features:
#tx = tx[:, 0:29]
# Mit de erste 29: 64.1%
# Mit aune ohni imputed:
#ix = [1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,29]
#tx = tx[:, ix]
# Max Acc = 61.616%


lambdaVec = np.logspace(-10, 0, base = 10, num = 50)
best_acc, best_lambda = cross_validation(y, tx, 10, lambdaVec, "svm",
                                         500000, 0.0001)

final_fit, loss = svm_classification(y, tx, best_lambda, np.zeros([tx.shape[1]]), 10000000, 0.0001)

# Predict on test
test = np.load("../../imputed/test_imputed.npy")

test = x_expand(test)
test = np.c_[np.ones(test.shape[0]), test]

test_preds = predict_svm_outcome(test, final_fit)
create_csv_submission(range(350000, 350000+test_preds.size), test_preds,
                      "../../submission/SVM_on_imputed_poly.csv")


