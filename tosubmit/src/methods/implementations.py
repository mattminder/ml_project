# Todo: TESTING!!!
# Todo: Document

import numpy as np

#Helpers
def linear_gradient(y, tx, w):
    "returns the mse gradient of the linear prediction"
    return -1/len(y)*np.dot(tx.T,y-np.dot(tx,w))

def linear_loss(y, tx, w):
    "computes the mean square error of the linear regression"
    return 1/2 * np.mean((np.dot(w,tx.T)-y)**2)


def logistic_prediction(tx, w):
    "returns the logistic regression prediction (probability of being a Higgs \
    boson based on the features and the trained weights"
    z = np.dot(tx,w)
    # Handle numerical problems (overflow and log(0))
    minimum = -np.log(np.finfo(z.dtype).max)
    maximum = -np.log(np.finfo(z.dtype).eps)
    z[z < minimum] = minimum
    z[z > maximum] = maximum
    return 1/(1+np.exp(-z))

def logistic_gradient(y, tx, w):
    "computes the gradient of the logistic regression"
    pred = logistic_prediction(tx,w)
    return -1/len(y)*np.dot(tx.T,y-pred)
                                            
def logistic_loss(y, tx, w):
    "computes the loss of the logistic regression"
    pred = logistic_prediction(tx,w)
    cost1 = -y*np.log(pred)
    cost2 = -(1-y)*np.log(1-pred)
    return np.mean(cost1+cost2)
     


#Implementation of the regression methods
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    "approximates weights of linear regression with gradient descent"
    w = initial_w
    for n_iter in range(max_iters):
        grad_L = linear_gradient(y,tx,w)
        w = w - gamma * grad_L
    loss = linear_loss(y,tx,w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    "approximates weights of linear regression with SGD based on one data point \
    at a time"
    w = initial_w
    for n_iter in range(max_iters):
        i = np.random.randint(len(y))
        grad_L = linear_gradient([y[i]], np.expand_dims(tx[i,:], axis=0),w)
        w = w - gamma * grad_L
    loss = linear_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    "computes weights of linear regression by solving the linear system"
    lhs = tx.T.dot(tx)
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs,rhs)
    loss = linear_loss(y,tx,w)
    return w, loss
    

def ridge_regression(y, tx, lambda_):
    "computes weights of ridge regression by solving the linear system"
    N = len(y)
    lhs = tx.T.dot(tx) + lambda_*2*N*np.eye(tx.shape[1])
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs,rhs)
    loss = linear_loss(y,tx,w) + lambda_*np.sum(np.sum(w**2))
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, verbose=False):
    "computes weights of logistic regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        if verbose & (n_iter % int(max_iters/20) == 0):
            print(str(int(n_iter / max_iters)*100) + "% Done")

        i = np.random.randint(len(y))
        grad_L = logistic_gradient([y[i]], np.expand_dims(tx[i,:], axis=0),w)
        w = w - gamma * grad_L
    loss = logistic_loss(y,tx,w)
    return w, loss
        
                                                                             
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    "computes weights of regularized (L2) logistic regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        i = np.random.randint(len(y))
        grad_L = logistic_gradient([y[i]], np.expand_dims(tx[i,:], axis=0),w) + lambda_*w
        w = w - gamma * grad_L
    loss = logistic_loss(y,tx,w) + lambda_/2*np.sum(np.sum(w**2))
    return w, loss

# Additional methods
def hinge_loss_gradient(y, tx, w):
    "Gradient of hinge loss function. Accepts either a single datapoint or a" 
    "number of datapoints and returns average"
    z = np.asarray(1-y*np.dot(tx,w))
    z[z<0] = 0
    
    hinge_grad = np.zeros(tx.shape)
    nonZero = np.argwhere(z != 0).flatten()
    if y.ndim == 0:
        y = np.expand_dims(y, axis=0)
    if tx.ndim == 1:
        tx = np.expand_dims(tx, axis=0)
    if hinge_grad.ndim == 1:
        hinge_grad = np.expand_dims(hinge_grad, axis=0)
    hinge_grad[nonZero,:] = -np.dot(y[nonZero],tx[nonZero,:])

    return np.mean(hinge_grad, axis=0)

def svm_classification(y, tx, lambda_, initial_w, max_iters, gamma):
    "Support vector machine classification"
    w = initial_w
    for n_iter in range(max_iters):
        rand_idx = np.random.randint(0,len(y))
        reg = lambda_*w
        reg[0] = 0
        grad_L = (hinge_loss_gradient(np.asarray(y[rand_idx]), tx[rand_idx,:], w) + reg)
        w = w - gamma * grad_L
    loss = 1-y*np.dot(tx,w)
    loss[np.asarray(loss)<0] = 0
    return w, np.mean(loss)

def predict_svm_outcome(tx, w):
    out = np.dot(tx,w)
    return np.sign(out)
    
