# APPLIES BEST SUBSET SELECTION TO POLY-EXPANDED SVM MODEL
from methods.implementations import *
from helpers.custom_helpers import *
import numpy as np
# -------------------------------
# LOADING
# Load data
tx = np.load("../../imputed/final_plus_dummy.npy")
y = np.load("../../imputed/y_train.npy")

# Discard dummies
tx = tx[:, :30]


# Polynomial expansion
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

# Definition of lambda_param, stepsize and niter
lambda_ = 4.498e-08
max_iters = 50000
gamma = 0.001


# Split into train and validation set (80/20 split)
perm = np.random.permutation(tx.shape[0])
splitIx = int(tx.shape[0]/5)
valSet = tx[perm[:splitIx], :]
trainSet = tx[perm[splitIx:], :]
valY = y[perm[:splitIx]]
trainY = y[perm[splitIx:]]


"""
# Forward BSS

# Result arrays with rows: step in BSS, Cols: Features
feature_included_fwd = np.zeros((30, 30))
computed_accuracy_fwd = np.zeros((30, 30))
max_accuracy_fwd = np.zeros(30)
for n_models in range(30):
    print("\n\nNumber Features: " + str(n_models))
    present_models = feature_included_fwd[n_models, :] > 0
    print(present_models)

    # Iterate through all models not included yet
    for ix in np.argwhere(present_models == 0):

        # Create sets for current iteration
        use_models = present_models == True # Copy array
        use_models[ix] = True
        tmp_train = trainSet[:, use_models]

        tmp_train = x_expand(tmp_train)
        tmp_train = np.c_[np.ones(tmp_train.shape[0]), tmp_train]

        tmp_val = valSet[:, use_models]
        tmp_val = x_expand(tmp_val)
        tmp_val = np.c_[np.ones(tmp_val.shape[0]), tmp_val]

        # Fit on train, predict on val
        fit, loss = svm_classification(trainY, tmp_train, lambda_,
                                       np.zeros(tmp_train.shape[1]), max_iters, gamma)
        pred = predict_svm_outcome(tmp_val, fit)
        acc = np.sum(pred == valY)/valY.size
        computed_accuracy_fwd[n_models, ix] = acc
        print("Add feature " + str(ix) + "Accuracy: " + str(acc))

    # Take model with highest accuracy for next iteration
    max_ix = np.argmax(computed_accuracy_fwd[n_models, :])
    max_acc = computed_accuracy_fwd[n_models, max_ix]
    max_accuracy_fwd[n_models] = max_acc
    feature_included_fwd[(n_models + 1):, max_ix] = 1

    print("\nMaximal accuracy: " + str(max_acc) + "\tincl feature " + str(max_ix))

# Save results
np.save("../../BSS/SVM_max_accuracy_fwd.npy", max_accuracy_fwd)
np.save("../../BSS/SVM_feature_included_fwd.npy", feature_included_fwd)
np.save("../../BSS/SVM_computed_accuracy_fwd.npy", max_accuracy_fwd)
"""



# Backward BSS

# Result arrays with rows: step in BSS, Cols: Features
feature_excluded_bwd = np.zeros((30, 30))
computed_accuracy_bwd = np.zeros((30, 30))
max_accuracy_bwd = np.zeros(30)
for n_excluded in range(30):
    print("\n\nNumber Excluded Features: " + str(n_excluded))
    present_models = feature_excluded_bwd[n_excluded, :] == 0
    print(present_models)

    # Exclude one of the present models
    for ix in np.argwhere(present_models):
        print("Remove feature " + str(ix))

        # Create sets for current iteration
        use_models = present_models == True # Copy array
        use_models[ix] = False
        tmp_train = trainSet[:, use_models]
        tmp_train = x_expand(tmp_train)
        tmp_train = np.c_[np.ones(tmp_train.shape[0]), tmp_train]

        tmp_val = valSet[:, use_models]
        tmp_val = x_expand(tmp_val)
        tmp_val = np.c_[np.ones(tmp_val.shape[0]), tmp_val]

        # Fit on train, predict on val
        fit, loss = svm_classification(trainY, tmp_train, lambda_,
                                       np.zeros(tmp_train.shape[1]), max_iters, gamma)
        pred = predict_svm_outcome(tmp_val, fit)
        acc = np.sum(pred == valY)/valY.size
        computed_accuracy_bwd[n_excluded, ix] = acc
        print("Accuracy: " + str(acc))

    # Take model with highest accuracy for next iteration
    max_ix = np.argmax(computed_accuracy_bwd[n_excluded, :])
    max_acc = computed_accuracy_bwd[n_excluded, max_ix]

    min_ix = np.argmin(computed_accuracy_bwd[n_excluded, :])
    min_acc = computed_accuracy_bwd[n_excluded, min_ix]

    max_accuracy_bwd[n_excluded] = max_acc
    feature_excluded_bwd[(n_excluded + 1):, max_ix] = 1

    print("\nMaximal accuracy: " + str(max_acc) + "\texcl feature " + str(max_ix))

# Save results
np.save("../../BSS/SVM_max_accuracy_bwd.npy", max_accuracy_bwd)
np.save("../../BSS/SVM_feature_excluded_bwd.npy", feature_excluded_bwd)
np.save("../../BSS/SVM_computed_accuracy_bwd.npy", max_accuracy_bwd)










