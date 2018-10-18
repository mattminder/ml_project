# Todo: TESTING!!!
# Todo: Document

import numpy as np


def compute_MSE_gradient(y, tx, w):
    return -1/len(y)*np.dot(tx.T,y-np.dot(tx,w))


# Todo: take as input only y and y hat, allows to generalize for other methods
def compute_MSE_loss(y, tx, w):
    return 1/2 * np.mean((np.dot(w,tx.T)-y)**2)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad_L = compute_MSE_gradient(y,tx,w)
        w = w - gamma * grad_L
    loss = compute_MSE_loss(y,tx,w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        for i in shuffle_indices:
            grad_L = compute_MSE_gradient(y[i], tx[i,:], w)
            w = w - gamma * grad_L
    loss = compute_MSE_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    lhs = tx.T.dot(tx)
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs,rhs)
    loss = compute_MSE_loss(y,tx,w)
    return w, loss


# Todo: Return loss as well (?)
# Todo: test performance against other solution (least_squares)
def multiple_least_squares(y, tx):
    xtx_inv = np.linalg.inv(tx.T.dot(tx))
    bhat = xtx_inv.dot(tx.T).dot(y)
    return bhat
    

def ridge_regression(y, tx, lambda_):
    N = len(y)
    lhs = tx.T.dot(tx) + lambda_*2*N*np.eye(tx.shape[1])
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs,rhs)
    loss = compute_MSE_loss(y,tx,w) + np.sum(np.sum(w**2))
    return w, loss


def logistic_prediction(tx, w):
    z = np.dot(tx,w)
    return 1/(1+np.exp(-z))
                                            

def logistic_gradient(tx, y, w):
    pred = logistic_prediction(tx,w)
    return 1/len(y)*np.dot(tx.T,y-pred)
                                            

def logistic_loss(tx, y, w):
    pred = logistic_prediction(tx,w)
    cost1 = -y*np.log(pred)
    cost2 = (1-y)*np.log(1-pred)
    return np.mean(cost1+cost2)
                                            

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad_L = logistic_gradient(y,tx,w)
        w = w - gamma * grad_L
    loss = logistic_loss(y,tx,w)
    return w, loss
                                                                                      

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad_L = logistic_gradient(y,tx,w) - lambda_*w
        w = w - gamma * grad_L
    loss = logistic_loss(y,tx,w) + np.sum(np.sum(w**2))
    return w, loss
                                            
                                         

