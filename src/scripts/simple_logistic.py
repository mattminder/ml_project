# Performs simple logistic regression
from methods.implementations import *
from helpers.custom_helpers import *
import numpy as np
from helpers.proj1_helpers import *


# Load data, augment with 1, convert result to 0/1
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")
y[np.where(y == -1)] = 0
tx_augmented = np.c_[np.ones(tx.shape[0]), tx]

# Fit model
logFit, logLoss = logistic_regression(y,
                                      tx,
                                      initial_w=np.ones(tx.shape[1])*.1,
                                      max_iters=1000,
                                      gamma=.1)

# Predict on Test Data
test = np.load("../../imputed/test_imputed.npy")
test_augmented = np.c_[np.ones(test.shape[0]), test]
test_predicted = logistic_prediction(test_augmented, logFit)
test_predicted[y == 0] = -1

# Create submission
create_csv_submission(test_predicted, range(test_predicted.size), "../../submission/simple_logistic_on_imputed.csv")
