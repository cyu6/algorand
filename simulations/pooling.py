import numpy as np
import bisect
from multiprocessing import Pool
from functools import partial
from constants import i, n, x, m

def hashplus_sample_pool(D, alpha, beta, l):
  sampler = partial(hashplus_wrapper, D, alpha, beta, l)

  F = np.zeros(n) # final distribution

  with Pool(4) as p:
    draws = p.map(sampler, F)

  return np.array(draws)

def hashplus_wrapper(D, alpha, beta, l, index):
  c = [0]*i # coin values
  r = [0]*i # (expected) reward values

  # step 1: draw adversary's coins
  # step 1a: draw c_1 from exp(alpha)
  c[0] = np.random.exponential(scale=(1/alpha))

  # step 1b: draw c_i for all i > 1
  for k in range(1, i):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))

  # step 2: for all i >= 1, draw r_i from D i.i.d.
  for k in range(i):
      r[k] = D[np.random.randint(low=0, high=n)]

  # step 3: output expectation over c0 and r0
  samples_sum = 0

  for _ in range(m):
    c0 = 500*(1/(1-alpha)) # very large number so it can't win
    if beta != 0:
      c0 = np.random.exponential(scale=(1/(beta*(1-alpha))))

    # let pos such that c_pos < c0 < c_pos+1
    pos_c = bisect.bisect_left(c, c0) if beta != 0 else i

    # compute g = adversary's best reward

    # TODO: optimize this with arange(pos_c)
    g = 0
    if pos_c != 0:
      g = np.max([np.exp(c[s]*(alpha-1)*(1-beta))*(1+r[s]) - x*s for s in range(pos_c)])

    # approximate P[r0 < r^h threshold]
    r_thresh = (g + x*pos_c) * np.exp((1-alpha)*(1-beta)*c0)
    k = np.count_nonzero(D < r_thresh)

    # approximate integral over PDF of r
    r_pdf = np.sum(np.where(D >= r_thresh, D*np.exp((alpha-1.0)*(1.0-beta)*c0) - x*pos_c, 0.0))
    
    # reset or no?
    reward = max(0.0, g*k + r_pdf)

    # add to sum
    samples_sum = samples_sum + reward
  
  return samples_sum/(m*n) - alpha - l
