import numpy as np
from constants import i, n

"""
Update rule for COIN strategy
- equivalent to the beta = 0 case, where the adversary knows nothing other than their own coins
- can be considered a lower bound on the rewards gained from this specific attack

Parameters:
D = distribution of expected future rounds won
alpha = proportional stake in system
"""
# Update rule for COIN strategy
def coin(D, alpha):
  c = [0]*i # coin values
  r = [0]*i # (expected) reward values
  F = [0]*n # final distribution

  for j in range(n):
    # step 1: draw c1 from exp(alpha)
    c[0] = np.random.exponential(scale=(1/alpha))

    # step 2: calculate ci for all i > 1
    for k in range(1, i):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))

    # step 3: for all i >= 1, draw r_i from D iid
    for k in range(i):
      r[k] = D[np.random.randint(low=0, high=n)]

    # step 4: output coin with maximum expected value
    exp_vals = np.multiply(np.exp(np.multiply(c, alpha-1)), np.add(r, 1))
    max_coin = np.max(exp_vals)

    F[j] = max_coin

  return F
