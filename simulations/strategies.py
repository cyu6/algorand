import numpy as np
import bisect
from constants import i, n, m, x

"""
Parameters:
D = distribution of expected future rounds won
alpha = proportional stake in system
beta = fraction of connectedness / knowledge of network, 0 < beta < 1
l = win l (lambda) fraction of rounds
"""

# Update rule for OPT strategy - equivalent to the beta = 1 case
def opt(D, alpha, _beta, _l):
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

    # step 4: output sum calculation 
    output_sum = np.exp(-1*(1-alpha)*c[0])
    best_so_far = r[0]
    for k in range(i-1):
      prob = np.exp(-1*(1-alpha)*c[k]) - np.exp(-1*(1-alpha)*c[k+1])
      max_reward = max(best_so_far, r[k])
      output_sum += prob*max_reward

      best_so_far = max_reward

    F[j] = output_sum

  return F


# Update rule for COIN strategy - equivalent to the beta = 0 case
def coin(D, alpha, _beta, _l):
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


def beta(D, alpha, beta, _l):
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

    # step 4: output sum calculation 
    output_sum = np.exp(-1*(1-alpha)*c[0])
    best_so_far = r[0]
    for k in range(i-1):
      prob = np.exp(-1*beta*(1-alpha)*c[k]) - np.exp(-1*beta*(1-alpha)*c[k+1])
      max_reward = max(best_so_far, r[k])
      output_sum += prob*max_reward

      best_so_far = max_reward

    F[j] = output_sum

  return F


# Update rule for OPTPLUS strategy - addition on to the OPT strategy by taking network's seed
def optplus(D, alpha, _beta, l):
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
      prob = np.exp(-1*(1-alpha)*c[k]) - np.exp(-1*(1-alpha)*c[k+1])
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


# Update rule for HASHPLUS strategy - calculate exact reward each round
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


# Update rule for HASHPLUS strategy - sample expected reward each round
def hashplus_sample(D, alpha, beta, l):
  c = [0]*i # coin values
  r = [0]*i # (expected) reward values
  F = np.zeros(n) # final distribution

  for j in range(n):
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
      g = 0
      if pos_c != 0:
        g = np.max([np.exp(c[s]*(alpha-1)*(1-beta))*(1+r[s]) - x*s for s in range(pos_c)])

      # approximate P[r0 < r^h threshold]
      r_thresh = (g + x*pos_c) * np.exp((1-alpha)*(1-beta)*c0)
      k = np.count_nonzero(D < r_thresh)

      # approximate integral over PDF of r
      r_pdf = np.sum(np.where(D >= r_thresh, D*np.exp((alpha-1.0)*(1.0-beta)*c0) - x*pos_c, 0.0))
      
      # add to sum
      samples_sum = samples_sum + (g*k + r_pdf)/n

    F[j] = samples_sum/m - alpha - l

  return F
