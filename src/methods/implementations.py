#

import numpy as np

def compute_MSE_gradient(y, tx, w):
  return -1/len(y)*np.dot(tx.T,y-np.dot(tx,w))

def compute_MSE_loss(y, tx, w):
  1/2 * np.mean((np.dot(w,tx.T)-y)**2)

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
      for y_batch, tx_batch in batch_iter(y,tx,batch_size=1):
        grad_L = compute_MSE_gradient(y_batch,tx_batch,w)
        w = w - gamma * grad_L
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    lhs = tx.T.dot(tx)
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs,rhs)
    loss = compute_MSE_loss(y,tx,w)
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    lhs = tx.T.dot(tx) + lambda_*2*N*np.eye(tx.shape[1])
    rhs = tx.T.dot(y)
    w = np.linalg.solve(lhs,rhs)
    loss = compute_MSE_loss(y,tx,w) + np.sum(np.sum(w**2))
    return w, loss
    
    
