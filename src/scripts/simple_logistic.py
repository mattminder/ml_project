# Performs simple logistic regression
from methods.implementations import *
from helpers.custom_helpers import *
import numpy as np
from helpers.proj1_helpers import *

# -------------------------------
# LOADING
# Load data, augment with 1, convert result to 0/1
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")
y[np.where(y == -1)] = 0

# -------------------------------
# NO DATA TRANSFORMATION
tx_augmented = np.c_[np.ones(tx.shape[0]), tx]

# Fit model
logFit, logLoss = logistic_regression(y,
                                      tx_augmented,
                                      initial_w=np.ones(tx_augmented.shape[1])*.001,
                                      max_iters=100,
                                      gamma=.1)

print("logLoss: " + str(logLoss))
print("Variable coefficients:")
print(logFit)


# Predict on Test Data, convert to -1/1 based on threshold = .5
test = np.load("../../imputed/test_imputed.npy")
test_augmented = np.c_[np.ones(test.shape[0]), test]
test_predicted = logistic_prediction(test_augmented, logFit)
test_predicted_out = test_predicted
test_predicted_out[test_predicted > .5] = 1
test_predicted_out[test_predicted <= .5] = -1


# Create submission
create_csv_submission(range(350000, 350000+test_predicted.size), test_predicted_out,
                      "../../submission/simple_logistic_on_imputed.csv")

# -------------------------------
# LOGARITHMIC DATA TRANSFORMATION
# Add constant > min(tx), then transform to log
tx_toLog = np.log(tx[:, 0:29] + 20)
tx_toLog_norm, features_mean, features_stdev = normalize_data(tx_toLog)
tx_log = np.column_stack((tx_augmented, tx_toLog_norm))

logFitLog, logLossLog = logistic_regression(y,
                                            tx_log,
                                            initial_w=np.ones(tx_log.shape[1])*.001,
                                            max_iters=100,
                                            gamma=.1)

print("logLoss: " + str(logLossLog))
print("Variable coefficients:")
print(logFitLog)


# Predict on Test Data, convert to -1/1 based on threshold = .5
test_toLog = np.log(test[:, 0:29] + 20)
test_toLog_norm = (test_toLog - features_mean)/features_stdev

test_log = np.column_stack((test_augmented, test_toLog_norm))
test_log_predicted = logistic_prediction(test_log, logFitLog)
test_log_predicted_out = test_log_predicted
test_log_predicted_out[test_log_predicted > .5] = 1
test_log_predicted_out[test_log_predicted <= .5] = -1


# Create submission
create_csv_submission(range(350000, 350000+test_log_predicted_out.size), test_log_predicted_out,
                      "../../submission/simple_logistic_on_imputed_log_transformed.csv")
