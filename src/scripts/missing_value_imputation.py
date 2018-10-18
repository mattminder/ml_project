"""Script for data loading and missing value imputation"""

from methods.implementations import *
from methods.proj1_helpers import *
import re

# Train Data loading
train = load_csv_data("../../all/train.csv")

# Separate 0 or 1
train_y = train[0]
train_x = train[1]

# Number obs and features
n_obs = train_x.shape[0]
n_feat = train_x.shape[1]

# For every observation create string with missing features
missing_features = []
missing_features_set = set()
for i in range(train_x.shape[0]):
    tmp = np.argwhere(train_x[i, :] == -999)
    toStr = np.array2string(tmp)
    prettyStr = re.sub("\]\\n \[", ":", toStr)
    prettyStr = re.sub("\[", "", prettyStr)
    prettyStr = re.sub("\]", "", prettyStr)
    prettyStr = re.sub(" ", "", prettyStr)
    missing_features.append(prettyStr)
    missing_features_set.add(prettyStr)
missing_features = np.array(missing_features)

print(str(len(missing_features_set)) + " different patterns of missing features.")

# Sort cells according to missing pattern
train_x_split = []
train_x_split_pattern = []
for pattern in missing_features_set:
    train_x_split.append(train_x[missing_features == pattern, :])
    train_x_split_pattern.append(pattern)

# Perform least squares for every set to impute missing values
# Todo: vilech übrprüefe obs würk ds full isch i index 0
full_set = train_x_split[0]

impute_fit = []
imputed = [full_set]
impute_loss = []

for i in range(1, len(train_x_split)):
    print(i)
    # Current subset to be fitted
    tmp = train_x_split[i]

    # Get indices of missing values
    missing = np.argwhere(tmp[1, :] == -999).flatten()
    present = (tmp[1, :] != -999).flatten()
    n_missing = missing.shape[0]

    # For every missing value, train least squares on full set with GD
    fit = np.empty(n_feat-n_missing, n_missing)
    loss = np.empty(n_missing)
    for j in range(n_missing):
        fit[:, j], loss[j] = least_squares_GD(y = full_set[:, missing[j]],
                                              tx= full_set[:, present],
                                              initial_w = np.ones(n_feat-n_missing),
                                              max_iters = 2000,
                                              gamma = .1)

    # Save fit and loss
    impute_fit.append(fit)
    impute_loss.append(loss)

    # Replace missing values with prediction
    prediction = fit.dot(tmp[:, present])
    tmp[:, missing] = prediction

    imputed.append(tmp)







# test = load_csv_data("../../all/test.csv")
