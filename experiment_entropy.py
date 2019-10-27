import argparse
import os
import numpy as np
import scipy.misc
import scipy
from copy import deepcopy
from scipy.linalg import solve_triangular
from lib import create_K, sinkhorn_loss, mog_loss



#computes the three entropy estimators. sink corresponds to h_{paired}. sink_ind corresponds to h_{ind} and MC corresponds to h_{m.g.} in the paper
#results are stored as a npy file on the entropy_variance folder
#each estimator is computed for n=100,500,1000,2000,5000,10000,15000, and for m =n *(1-lambda)/lambda, where lambda=[0.3,0.5,0.7] (for paper figures only lambda =0.5 is relevant)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--d', type=int, default=5, help='sequencd length')
parser.add_argument('-n_sim', '--n_sim', type=int, default=20, help='sequencd length')
parser.add_argument('-n_chain', '--n_chain', type=int, default=5, help='sequencd length')
parser.add_argument('-noise_level', '--noise_level', type=float, default=1.0, help='sequencd length')
parser.add_argument('-noise', '--noise', type=float, default=1.0, help='sequencd length')

args = parser.parse_args()


n_chain = args.n_chain #chain number
d = args.d #dimension
n_sim = args.n_sim #number of samples
noise = args.noise +0.0 #base noise (distribution P)
noise_level = args.noise_level +0.0 #gaussian noise

dir = './Results/'

try:
  os.mkdir(dir)
except:
  1

n_samples_max = 15000
n_samples_max2 = int(15000*0.7/0.3)

n_samples_vec = [100,500,1000,2000,5000,10000, 15000]

ll = [0.3, 0.5, 0.7]
MC = np.zeros((len(ll),len(n_samples_vec), n_sim))
sink = np.zeros((len(ll),len(n_samples_vec), n_sim))
sink_ind = np.zeros((len(ll),len(n_samples_vec), n_sim))

for i in range(n_sim):

  mu0 = np.tile(np.array([-1, 1]), [d, 1]).T
  b = np.random.binomial(1, 0.5, [1, n_samples_max])
  b2 = np.random.binomial(1, 0.5, [1, n_samples_max2])
  mu = np.reshape(mu0[b], [-1, d])
  mu2 = np.reshape(mu0[b2], [-1,d])

  x_real = np.random.normal(0.0, 1.0, (n_samples_max, d)) * np.sqrt(noise) +mu
  x_real2 = np.random.normal(0.0, 1.0, (n_samples_max2, d)) * np.sqrt(noise) +mu2
  noise_vector = np.random.normal(0, 1, [n_samples_max, d]) * np.sqrt(noise_level)
  noise_vector2 = np.random.normal(0, 1, [n_samples_max2, d]) * np.sqrt(noise_level)

  y_real2 = noise_vector2 + x_real2
  y_real = noise_vector + x_real
  K = create_K(x_real, y_real) / (2 * (noise_level))
  K2 = create_K(x_real, y_real2) / (2 * (noise_level))

  for k in range(len(n_samples_vec)):
    for l0 in range(len(ll)):
      print([i,k,l0])
      l = ll[l0]
      n_samples = n_samples_vec[k]
      n_mcmc = int(n_samples * (1 - l) / l)
      print(n_mcmc)
      ss = sinkhorn_loss(K[:n_samples, :n_samples], n_iter=20)
      ss2 = sinkhorn_loss(K2[:n_samples, :n_samples], n_iter=20)

      ssm = mog_loss(K)

      MC_val = -(ssm) + 0.5 * d * np.log(2.0 * np.pi * noise_level)

      sinkhorn_val = (ss) + 0.5 * d * np.log(2.0 * np.pi * noise_level)
      sinkhorn_val_ind = (ss2) + 0.5 * d * np.log(2.0 * np.pi * noise_level)

      MC[l0,k, i] = MC_val
 
      sink[l0,k, i] = sinkhorn_val
      sink_ind[l0,k, i] = sinkhorn_val_ind

dictionary = {'sink': sink, 'sink_ind': sink_ind, 'MC': MC,
              'noise':noise, 'noise_level':noise_level,'d':d}
print(dictionary)
np.save(dir + 'c='+str(n_chain) +  ',noise_level=' + str(noise_level) + ',noise_base=' +str(noise) +',d=' +str(d)+'.npy', dictionary)
