import numpy as np
import statistics

"""Strategies"""
from strategies import opt, coin, optplus, hashplus_exact, hashplus_sample
from pooling import hashplus_sample_pool

"""Setup"""
from constants import i, n, m, x

# Simulates expected rewards for given alpha and lambda.
def simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag):
  D_honest = np.full(n, (-l-alpha))
  # D_honest = np.zeros(n)
  
  # run first iteration
  F0 = strategy_sim(D_honest, alpha, beta, l)

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
    nowF = strategy_sim(bestF, alpha, beta, l)
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

# Run simulation for alpha = 0.1 to 1.0 in increments of 0.01.
def gather_data(strategy_sim, beta, difference, times, output_file_name, debug_flag):
  distributions = np.zeros((100, n+1))

  for num in range(1, 100): # multiples of 1/100
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
      if abs(reward) < 0.01:
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
  # single_alpha_run(alpha=0.01, beta=0.5, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True)

  # single_alpha_run(alpha=0.1, beta=0.5, strategy_sim=hashplus_exact, difference=0.01, times=4, debug_flag=True)

  # attempt at sanity check
  # HASHPLUS_SAMPLE
  # single_alpha_run(alpha=0.1, beta=0, strategy_sim=hashplus_sample, difference=0.001, times=2, debug_flag=True, end_lambda=0)
  # COIN
  # single_alpha_run(alpha=0.1, beta=0, strategy_sim=coin, difference=0.01, times=4, debug_flag=True)
 
 
  gather_data(hashplus_sample_pool, beta=0.25, difference=0.01, times=4, output_file_name="HASHPLUS_trial=2_beta=0.5_alpha=0.01-1.0.npy", debug_flag=True)

  # distributions = np.load("OPTPLUS distributions 0.01 to 1.0.np32y")
  # gather_data_from_given_start(distributions, optplus_sim, start_alpha=0.86, 
  #   difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)


if __name__ == "__main__":
    main()

