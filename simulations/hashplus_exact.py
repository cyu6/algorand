import bisect
import numpy as np
from constants import i, n, x

"""
Update rule for HASHPLUS strategy, calculate exact reward each round
- adversary has more information about the network than they should
- this allows us to calculate the exact expected reward though

Parameters:
D = distribution of expected future rounds won
alpha = proportional stake in system
beta = fraction of connectedness / knowledge of network, 0 < beta < 1
l = win l (lambda) fraction of rounds
"""
def hashplus_exact(D, alpha, beta, l):
  c = [0]*i # coin values
  r = [0]*i # (expected) reward values
  F = [0]*n # final distribution

  for j in range(n):
    # step 1: draw adversary's coins
    # step 1a: draw c_1 from exp(alpha)
    c[0] = np.random.exponential(scale=(1/alpha))

    # step 1b: calculate c_i for all i > 1
    for k in range(1, i):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))
    
    # step 2: for all i >= 1, draw r_i from D i.i.d.
    for k in range(i):
      r[k] = D[np.random.randint(low=0, high=n)]
    
    # step 3: draw network's coin and corresponding reward
    c0 = 0
    r0 = 0
    if beta != 0:
      c0 = np.random.exponential(scale=(1/(beta*(1-alpha))))
      r0 = D[np.random.randint(low=0, high=n)]

    # step 4: let pos such that c_pos < c0 < c_pos+1
    pos =  bisect.bisect_left(c, c0) if beta != 0 else i

    # step 5: output reward. note: if we take the honest coin, we also subtract l to "reset".
    # step 5a: calculate reward if we take the honest coin
    loss_reward = np.exp((alpha-1)*(1-beta)*c0) * r0 - x*i

    # step 5b: if pos > 0, determine best coin out of adversary's winners and compare with honest
    win_reward = 0
    if pos != 0:
      win_reward = np.max([np.exp(c[s]*(alpha-1)*(1-beta))*(1+r[s]) - x*(s-1) for s in range(pos)])

    output = win_reward if win_reward > loss_reward else loss_reward

    # print("output: ", output)
    F[j] = output - l - alpha

  return F
