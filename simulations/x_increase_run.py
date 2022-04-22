import numpy as np
import bisect
import scipy.special as sc
from multiprocessing import Pool
from functools import partial
from constants import i, n, m

def hashplus_sample_pool(D, alpha, beta, l, x):
  sampler = partial(hashplus_wrapper, D, alpha, beta, l, x)

  F = np.zeros(n) # final distribution

  with Pool(4) as p:
    draws = p.map(sampler, F)

  return np.array(draws)

def hashplus_wrapper(D, alpha, beta, l, x, index):
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
    reward = max(0.0-x*pos_c, g*k + r_pdf)

    # add to sum
    samples_sum = samples_sum + reward
  
  return samples_sum/(m*n) - alpha - l

# Simulates expected rewards for given alpha and lambda.
def simulate(alpha, beta, l, x, strategy_sim, difference, times, debug_flag):
  expected_winners = np.sum([sc.betainc(float(j), 1, alpha/(alpha + beta*(1-alpha))) for j in range(1, i+1)])
  D_honest = np.full(n, (-l-alpha-x*expected_winners))
  
  # run first iteration
  F0 = strategy_sim(D_honest, alpha, beta, l, x)

  if debug_flag:
    print("FIRST ROUND EXPECTED REWARDS: ", np.average(F0))

  # run sim until expectation doesn't change by _difference_ for _times_ times in a row
  bestF = F0
  lastavg = np.average(F0)
  currentavg = 0
  runs = 1
  runs_in_range = 0
  while abs(lastavg - currentavg) > difference or runs_in_range < times:
  # const = 0
  # while const < 10:
    nowF = strategy_sim(bestF, alpha, beta, l, x)
    lastavg = np.average(bestF)
    currentavg = np.average(nowF)
    bestF = nowF
    runs += 1

    if abs(lastavg - currentavg) < difference:
      runs_in_range += 1
    elif abs(lastavg - currentavg) > difference and runs_in_range > 0:
      runs_in_range = 0

    # const += 1
    
    if debug_flag:
      print("RUN #: ", runs)
      print("CURR AVG: ", currentavg)
      print("RUN IN RANGE #: ", runs_in_range)
      # print("VARIANCE: ", statistics.variance(nowF))

  return bestF

def single_alpha_run(alpha, beta, strategy_sim, difference, times, debug_flag, end_lambda = 1):
  if debug_flag: print("Alpha: ", alpha)

  # binary search for lambda
  l = 0
  start = 0
  end = end_lambda

  while (start <= end):
    l = (start + end)/2
    if debug_flag: print("Lambda: ", l)

    bestF = simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag)
    reward = np.average(bestF)

    if debug_flag: print("Reward for ", l, ": ", reward)
    
    # arbitrary error diff
    if abs(reward) < 0.001:
      break

    if start == end: break

    if reward > 0:
      start = l
    else:
      end = l

  if debug_flag: print("LAST lambda was ", l)

# Run simulation for x = 0 to 10 in increments of 0.1.
def gather_data_for_x_fixed_alpha_beta(strategy_sim, alpha, beta, difference, times, output_file_name, debug_flag):
  distributions = np.zeros((101, n+2))

  for num in range(0, 101): # multiples of 1/100
    x = num/10
    if debug_flag: print("x reward: ", x)

    # binary search for lambda
    l = 0  # same as mid
    start = 0
    end = 1

    while (start <= end):
      l = (start + end)/2
      if debug_flag: print("Lambda: ", l)

      bestF = simulate(alpha, beta, l, x, strategy_sim, difference, times, debug_flag)
      reward = np.average(bestF)

      if debug_flag: print("Reward for ", l, ": ", reward)

      # arbitrary error diff
      if abs(reward) < 0.001:
        break

      if start == end: break

      if reward > 0:
        start = l
      else:
        end = l
      
    if debug_flag: print("LAST lambda was ", l)

    # Save the best distribution and corresponding lambda
    distributions[num][0] = x
    distributions[num][1] = l
    distributions[num][2:] = bestF
    np.save(output_file_name, distributions)

# Run simulation starting at alpha = start_alpha to 1.0 in increments of 0.01.
def gather_data_from_given_start(distributions, strategy_sim, start_alpha, beta, difference, times, output_file_name, debug_flag):  
  for num in range(int(start_alpha*100), 100): # multiples of 1/100
    alpha = num / 100
    
    if debug_flag: print("ALPHA: ", alpha)
    
    # binary search for lambda
    l = 0  # same as mid
    start = 0
    end = 1

    while (start <= end):
      l = (start + end)/2
      if debug_flag: print("Lambda: ", l)

      bestF = simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag)
      reward = np.average(bestF)

      if debug_flag: print("Reward for ", l, ": ", reward)
      
      # arbitrary error diff
      if abs(reward) < 0.001:
        break

      if start == end: break

      if reward > 0:
        start = l
      else:
        end = l
      
    if debug_flag: print("LAST lambda was ", l)

    # Save the best distribution and corresponding lambda
    distributions[num-1][0] = l
    distributions[num-1][1:] = bestF
    np.save(output_file_name, distributions)



def main():
  # distributions = np.zeros((100, n+1))
  # distributions = np.load("OPTPLUS distributions 0.01 to 1.0.np32y")
  # gather_data_from_given_start(distributions, hashplus_sample_pool, start_alpha=0.09, beta=0.5,
  #  difference=0.01, times=4, output_file_name="HASHPLUS_x=0.01_beta=0.5_alpha=0.01-1.0.npy", debug_flag=True)

  gather_data_for_x_fixed_alpha_beta(hashplus_sample_pool, alpha=0.25, beta=1.0, difference=0.01, times=4, output_file_name="X_INCREASE_0.1-10_alpha=0.25_beta=1", debug_flag=True)

if __name__ == "__main__":
    main()


# LAST lambda was  0.087890625 -> first run for x = 0
# LAST lambda was  0.0947265625 -> second run for x = 0