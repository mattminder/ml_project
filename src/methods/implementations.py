# Todo: TESTING!!!
# Todo: Document
# Todo: Logistic regression formle überprüefe!!!

import numpy as np

#Helpers
def linear_gradient(y, tx, w):
    "returns the mse gradient of the linear prediction"
    return -1/len(y)*np.dot(tx.T,y-np.dot(tx,w))

def linear_loss(y, tx, w):
    "computes the mean square error of the linear regression"
    return 1/2 * np.mean((np.dot(w,tx.T)-y)**2)


def logistic_prediction(tx, w):
    "returns the logistic regression prediction based on the features and the \
    trained weights"
    z = np.dot(tx,w)
    return 1/(1+np.exp(-z))

def logistic_gradient(y, tx, w):
    "computes the gradient of the logistic regression"
    pred = logistic_prediction(tx,w)
    return 1/len(y)*np.dot(tx.T,y-pred)
                                            
def logistic_loss(y, tx, w):
    "computes the loss of the logistic regression"
    pred = logistic_prediction(tx,w)
    cost1 = -y*np.log(pred)
    cost2 = (1-y)*np.log(1-pred)
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
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        for i in shuffle_indices:
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
    loss = linear_loss(y,tx,w) + np.sum(np.sum(w**2))
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    "computes weights of logistic regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        for i in shuffle_indices:
            grad_L = logistic_gradient([y[i]], np.expand_dims(tx[i,:], axis=0),w)
            w = w - gamma * grad_L
    loss = logistic_loss(y,tx,w)
    return w, loss
        
                                                                             
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    "computes weights of regularized (L2) logistic regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        for i in shuffle_indices:
            grad_L = logistic_gradient([y[i]], np.expand_dims(tx[i,:], axis=0)) - lambda_*w
            w = w - gamma * grad_L
    loss = logistic_loss(y,tx,w) + np.sum(np.sum(w**2))
    return w, loss

