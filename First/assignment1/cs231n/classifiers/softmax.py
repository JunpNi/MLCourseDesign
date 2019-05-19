import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    #prevent numeric instability
  sm_score = sm_exp = np.zeros(10)
  
  for i in range(N):
    score = X[i].dot(W)
    score -= np.max(score)
    sm_score = np.exp(score)
    for j in range(C):
      sm_exp[j] = sm_score[j] / np.sum(sm_score)
      if j == y[i]:
        dW[:,j] += (-1 + sm_exp[j]) * X[i]
      else:
        dW[:,j] += sm_exp[j] * X[i]
      
        
    loss_set = -np.log(sm_exp[y[i]])
    loss += loss_set
  
  loss = loss/X.shape[0] + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #since the deriviation is Pi - 1/N, where Pi is the exp/sum_exp
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores)  #prevent numeric instability
  sm_scores = sm_exp = np.zeros_like(scores)
  loss_set = np.zeros(N)
  sm_scores = np.exp(scores)
  sm_sum_scores = np.sum(sm_scores, axis = 1)
  true_scores = scores[range(N),y]
  sm_exp = sm_scores / sm_sum_scores.reshape([N,1])

  loss = np.sum(np.log(sm_sum_scores)) - np.sum(true_scores)
    
  #for i in range(500):
     #margins = np.maximum(0,scores[i]-scores[i][y[i]]+1)
     #margins[y[i]] = 0
     #loss_set[i] = np.sum(margins)
  #Loss_svm = np.sum(loss_set)/N
  for i in range(N):
     dW += sm_exp[i] * X[i].reshape(D,1)   #this step is doing the j loop in naive func
                                #with broadcast attribute
        # [a,b,c] * [[1],[2]] = [[a,b,c],[2a,2b,2c]]
        #equal to [[a,b,c],[a,b,c]] * [[1,1,1],[2,2,2]]
     dW[:,y[i]] -= X[i]
    
  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

