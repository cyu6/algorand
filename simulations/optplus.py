import math
import numpy as np
from constants import i, n

"""
Update rule for OPTPLUS strategy
- addition on to the OPT strategy
- network's coin may give adversary larger rewards than any of the adversary's own winners
- in order to compare long term rewards, must use lambda param to compute fraction of rounds won by adversary

Parameters:
D = distribution of expected future rounds won
alpha = proportional stake in system
l = win l (lambda) fraction of rounds
"""
def optplus(D, alpha, l):
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
    
    # step 4: draw reward for taking network's coin as seed
    r0 = D[np.random.randint(low=0, high=n)]

    # step 5: output reward
    output_sum = 0
    best_so_far = r[0]
    all_winners = True
    for k in range(i-1):
      prob = math.exp(-1*(1-alpha)*c[k]) - math.exp(-1*(1-alpha)*c[k+1])
      max_reward = max(best_so_far, r[k])
      opt_reward = max(max_reward+1-l, r0-l)
      if opt_reward == (r0-l):
        all_winners = False
        # output_sum -= l
        # print("Losing ", opt_reward, " vs Winning ", max_reward+1-l)
      output_sum += prob*opt_reward

      best_so_far = max_reward

    output_sum = output_sum-l if all_winners else output_sum
    # if not all_winners:
    #   output_sum -= l
    #   print("subtracted lambda")

    F[j] = output_sum

  return F
