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
impute_fit = []
imputed = [train_x_split[0]]
for i in range(1, len(train_x_split)):
    print(i)
    # Current subset to be fitted
    tmp = train_x_split[i]

    # Get indices of missing values
    missing = np.argwhere(tmp[1, :] == -999)
    present = np.argwhere(tmp[1, :] != -999)

    # Fit model based on set without missing values
    # Todo: Doesnt work yet, dimension error in least squares function
    fit, loss = least_squares(y=train_x_split[0][:, missing],
                              tx=train_x_split[0][:, present])
    impute_fit.append(fit)
    prediction = fit.dot(tmp[:, present])

    # Replace missing values with prediction
    tmp[:, missing] = prediction

    imputed.append(tmp)







# test = load_csv_data("../../all/test.csv")
