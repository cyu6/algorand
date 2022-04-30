import numpy as np
import bisect
import scipy.special as sc
from constants import i, n, m
from hashplus_pooling import hashplus_sample_pool

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

def single_alpha_x_run(alpha, beta, x, strategy_sim, difference, times, debug_flag, end_lambda = 1):
  if debug_flag: print("x reward: ", x)

  # binary search for lambda
  l = 0
  start = 0
  end = end_lambda

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

# Run simulation for x = 0 to 10 in increments of 0.1.
def gather_data_for_x_fixed_alpha_beta(strategy_sim, alpha, beta, difference, times, output_file_name, debug_flag):
  distributions = np.zeros((100, n+2))
  for num in range(100): # multiples of 1/10
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

def main():
  # gather_data_for_x_fixed_alpha_beta(hashplus_sample_pool, alpha=0.25, beta=1.0, difference=0.01, times=4, output_file_name="X_INCREASE_7.5-11_alpha=0.25_beta=1", debug_flag=True)

  single_alpha_x_run(0.25, 1, 100, hashplus_sample_pool, 0.01, 4, True)


if __name__ == "__main__":
    main()


# LAST lambda was  0.087890625 -> first run for x = 0
# LAST lambda was  0.0947265625 -> second run for x = 0

#[0.0751953125, 0.0751953125, 0.0751953125, 0.0751953125, 0.0751953125, 0.0751953125]
#[4.5, 5.0, 5.5, 6.0, 6.5, 7.0]