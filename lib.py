import numpy as np
from time import time
import scipy.misc
import scipy
from copy import deepcopy
from scipy.linalg import solve_triangular

def sinkhorn_loss(K,  n_iter):
  #compute sinkhorn loss
  mu_x = np.ones(K.shape[0])/K.shape[0]
  mu_y = np.ones(K.shape[1])/K.shape[1]

  v = np.ones(len(mu_y))
  K_exp = np.exp(-1 * K)

  for _ in range(n_iter):

    u = np.divide(mu_x, np.dot(K_exp, v))
    v = np.divide(mu_y, np.dot(K_exp.T, u))
  u = np.divide(mu_x, np.dot(K_exp, v))
  val1 = np.sum(np.matmul(mu_x, np.log(u / mu_x)))
  val2 = np.matmul(np.log(v / mu_y), mu_y)
  val = val1+val2

  return val


def create_K(x_real, y_real):
  #create K matrix, useful for sinkhorn loss
  N_x = x_real.shape[0]
  N_y = y_real.shape[0]

  normx = np.tile(np.sum(x_real ** 2, 1, keepdims=True),  [1, N_y])
  normy = np.tile(np.sum(y_real **2 , 1, keepdims=True).T, [N_x, 1])

  z = np.matmul(x_real, y_real.T)
  return (normx - 2 * z + normy)



def mog_loss(K):
  #computes mog loss
  return np.mean(scipy.special.logsumexp(-K, 0)) - np.log(K.shape[0])


def reshape12(array):
  # collapses first and second axis
  s = np.shape(array)

  newshape = np.ones(len(s) - 1)
  newshape[0] = s[0] * s[1]
  for i in range(len(s) - 2):
    newshape[i + 1] = s[i + 2]
  return np.reshape(array, newshape.astype(int))


def bootstrap(array, B):
  # boostrap samples of the mean of an array
  b = []
  for i in range(B):
    barray = array[np.random.choice(len(array), len(array), replace=True)]
    b.append(np.nanmean(barray))
  return np.array(b)


def mixture_gaussians(mu, sigma, x, power=1.0, constant=0):
  d = x.shape[1]
  N_x = x.shape[0]
  N_mu = mu.shape[0]

  normx = np.tile(np.reshape(np.sum(x ** 2, 1), [N_x, -1]), [1, N_mu])
  normu = np.tile(np.reshape(np.sum(mu ** 2, 1), [-1, N_mu]), [N_x, 1])
  z = np.matmul(np.reshape(x, [N_x, -1]), mu.T)
  K = -1 * (normx - 2 * z + normu) / (2 * sigma)

  ldif = scipy.special.logsumexp(K - constant, 1)

  return np.mean(ldif ** power)


def entropy_mixture(mu, sigma, n_int=1, power=1.0, constant=0):
  l = np.zeros(mu.shape[0])
  d = mu.shape[1]
  constant = mu.shape[1] * 0.5 * np.log(2.0 * np.pi * sigma) + np.log(mu.shape[0])
  for j in range(len(l)):
    xx = np.random.normal(0, 1.0, (n_int, d)) * np.sqrt(sigma) + np.tile(mu[j, :], [n_int, 1])
    l[j] = np.mean(mixture_gaussians(mu, sigma, xx, power, constant))

  return -1 * np.mean(l)
