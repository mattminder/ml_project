"""Script for data loading and missing value imputation"""
#--------------------------------------
# HEADER
from methods.implementations import *
from methods.proj1_helpers import *
from helpers.custom_helpers import *
import re
import pickle

# Global vars
MISSING = -999
MAX_ITERS = 2000
GAMMA = .03
#--------------------------------------
# SCRIPT
# Train Data loading
print("Load train")
train = load_csv_data("../../all/train.csv")
print("Load test")
test = load_csv_data("../../all/test.csv")
print("Done loading")

# Separate 0 or 1
train_y = train[0]
train_x_unnorm = train[1]
test_x_unnorm = test[1]

# Number obs and features
n_obs = train_x_unnorm.shape[0]
n_obs_test = test_x_unnorm.shape[0]
n_feat = train_x_unnorm.shape[1]

# Normalization
train_x, features_mean, features_stdev = normalize_data(train_x_unnorm)
train_x[np.isnan(train_x)] = MISSING

test_x_unnorm[test_x_unnorm == -999] = np.nan
test_x = (test_x_unnorm-features_mean)/features_stdev
test_x[np.isnan(test_x)] = MISSING




# For every observation create string with missing features
missing_features = []
missing_features_set = set()
for i in range(train_x.shape[0]):
    tmp = np.argwhere(train_x[i, :] == MISSING)
    toStr = np.array2string(tmp)
    prettyStr = re.sub("\]\\n \[", ":", toStr)
    prettyStr = re.sub("\[", "", prettyStr)
    prettyStr = re.sub("\]", "", prettyStr)
    prettyStr = re.sub(" ", "", prettyStr)
    missing_features.append(prettyStr)
    missing_features_set.add(prettyStr)
missing_features = np.array(missing_features)

print("Train: " + str(len(missing_features_set)) + " different patterns of missing features.")

# Same for test set
missing_features_test = []
missing_features_set_test = set()
for i in range(test_x.shape[0]):
    tmp = np.argwhere(test_x[i, :] == MISSING)
    toStr = np.array2string(tmp)
    prettyStr = re.sub("\]\\n \[", ":", toStr)
    prettyStr = re.sub("\[", "", prettyStr)
    prettyStr = re.sub("\]", "", prettyStr)
    prettyStr = re.sub(" ", "", prettyStr)
    missing_features_test.append(prettyStr)
    missing_features_set_test.add(prettyStr)
missing_features_test = np.array(missing_features_test)

print("Test: " + str(len(missing_features_set_test)) + " different patterns of missing features.")



# Create dummy variables for each pattern of missing observations
missing_dummy = np.empty((n_obs, len(missing_features_set)))
missing_dummy_test = np.empty((n_obs_test, len(missing_features_set)))
missing_dummy_names = []
ix = 0
for pattern in missing_features_set:
    missing_dummy[:, ix] = missing_features == pattern
    missing_dummy_test[:, ix] = missing_features_test == pattern
    missing_dummy_names.append(pattern)
    ix += 1

# Remove dummy variables for complete, fully determined by other dummies
missing_dummy = np.delete(missing_dummy, 0, axis=1)
missing_dummy_test = np.delete(missing_dummy_test, 0, axis=1)
del missing_dummy_names[0]



# Sort cells according to missing pattern
train_x_split = []
test_x_split = []

train_x_split_pattern = []
for pattern in missing_features_set:
    train_x_split.append(train_x[missing_features == pattern, :])
    test_x_split.append(test_x[missing_features_test == pattern, :])
    train_x_split_pattern.append(pattern)



# Perform least squares for every set to impute missing values
full_set = train_x_split[0]
assert(np.all(full_set != MISSING)) # Verify that full_set doesn't have NA

impute_fit = []
imputed = [full_set]
imputed_test = [test_x_split[0]]

final = train_x
final_test = test_x

impute_loss = []

for i in range(1, len(train_x_split)):
    print("Pattern number: " + str(i))
    # Current subset to be fitted
    tmp = train_x_split[i]
    tmp_test = test_x_split[i]

    # Get indices of missing values
    missing = np.argwhere(tmp[0, :] == MISSING).flatten()
    present = (tmp[0, :] != MISSING).flatten()
    n_missing = missing.shape[0]

    # Create intermediary tx, augment with 1
    tx = full_set[:, present]
    tx = np.c_[ np.ones(tx.shape[0]), tx ]

    # For every missing value, train least squares on full set with GD
    fit = np.empty((tx.shape[1], n_missing))
    loss = np.empty(n_missing)
    for j in range(n_missing):

        fit[:, j], loss[j] = least_squares_GD(y = full_set[:, missing[j]],
                                              tx = tx,
                                              initial_w = np.ones(tx.shape[1]),
                                              max_iters = MAX_ITERS,
                                              gamma = GAMMA)
        print("Missing column nr. " + str(j) + " of " + str(n_missing) +
              ". Loss = " + str(loss[j]))

    # Save fit and loss
    impute_fit.append(fit)
    impute_loss.append(loss)

    # Replace missing values with prediction
    tmp_aug_present = np.c_[np.ones(tmp.shape[0]), tmp[:, present]]
    test_augmented = np.c_[np.ones(tmp_test.shape[0]), tmp_test[:, present]]

    prediction = tmp_aug_present.dot(fit)
    prediction_test = test_augmented.dot(fit)
    tmp[:, missing] = prediction
    tmp_test[:, missing] = prediction_test

    imputed.append(tmp)
    imputed_test.append(tmp_test)

    # Replace missing values in final set
    tmp_rep = final[missing_features == train_x_split_pattern[i], :]
    tmp_rep[:, missing] = prediction
    final[missing_features == train_x_split_pattern[i], :] = tmp_rep

    tmp_rep = final_test[missing_features_test == train_x_split_pattern[i], :]
    tmp_rep[:, missing] = prediction_test
    final_test[missing_features_test == train_x_split_pattern[i], :] = tmp_rep




# Bind final array
final_plus_dummy = np.column_stack((final, missing_dummy))
test_plus_dummy = np.column_stack((final_test, missing_dummy_test))




# Save
np.save("../../imputed/final_plus_dummy.npy", final_plus_dummy)
np.save("../../imputed/test_imputed.npy", test_plus_dummy)

np.save("../../imputed/y_train.npy", train_y)
np.save("../../imputed/ids_test.npy", test[2])

with open("../../imputed/impute_loss_pickle.txt", "wb") as fp:
    pickle.dump(impute_loss, fp)
with open("../../imputed/impute_fit_pickle.txt", "wb") as fp:
    pickle.dump(impute_fit, fp)
with open("../../imputed/missing_dummy_names.txt", "wb") as fp:
    pickle.dump(missing_dummy_names, fp)

#with open("../../imputed/loss_pickle.txt", "rb") as fp:   # Unpickling
#    b = pickle.load(fp)

# test = load_csv_data("../../all/test.csv")
