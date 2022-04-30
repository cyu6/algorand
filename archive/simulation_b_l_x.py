import bisect
import math
from cv2 import sort
import numpy as np
from scipy import misc
from sqlalchemy import all_, false, true

"""Setup"""

# Number of coins
i = 100
# Length of distribution
n = 10000
# Samples of network's coin
m = 10
# Reward for each hash reveal
x = 0

"""
Update rule for OPTPLUS strategy

Parameters:
D = distribution of expected future rounds won
alpha = proportional stake in system
beta = fraction of connectedness / knowledge of network, 0 < beta < 1
l = win l (lambda) fraction of rounds
"""
def optplus_sim(D, alpha, beta, l):
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
      c0 = np.random.exponential(scale=(1/(beta*(1-alpha))))

      # let pos such that c_pos < c0 < c_pos+1
      pos_c =  bisect.bisect_left(c, c0)

      # compute g = adversary's best reward
      g = 0
      if pos_c != 0: 
      #   for s in range(pos_c):
      #     c_reward = np.exp(c[s]*(alpha-1)*(1-beta))*(1+r[s]) - x*(s-1)
      #     g = c_reward if c_reward > g else g
        g = np.max([np.exp(c[s]*(alpha-1)*(1-beta))*(1+r[s]) - x*(s-1) for s in range(pos_c)])
      # g = np.max(rewards)
      # print("G: ", g)

      # approximate P[r0 < r^h threshold]
      r_thresh = (g + x*pos_c) * np.exp((1-alpha)*(1-beta)*c0)
      # print("rt: ", r_thresh)
      # sorted_D = sorted(D)
      # pos_D =  bisect.bisect_left(sorted_D, r_thresh)
      # C = pos_D / n
      # print("posD: ", pos_D)
      # print("sel: ", selected)
      numerator = np.count_nonzero(D < r_thresh)
      C = numerator/n
      # print("value for C: ", C)

      # approximate integral over PDF of r
      selected = np.where(D > r_thresh, D*np.exp((alpha-1.0)*(1.0-beta)*c0) - x*pos_c, 0.0)
      r_pdf = np.sum(selected)/n
      # print(r_pdf)

      # add to sum
      # print("in loop ss: ", samples_sum)
      # print(g)
      # print(C)
      # print(r_pdf)
      samples_sum = samples_sum + g*C + r_pdf

    # print("out loop ss: ", samples_sum)
    F[j] = samples_sum/m - alpha - l
    # print("output: ", F[j])

  return F

# Simulates expected rewards for given alpha and lambda.
def simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag):
  D_honest = np.zeros(n)
  
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
    nowF = strategy_sim(bestF, alpha, beta, l)
    lastavg = np.average(bestF)
    currentavg = np.average(nowF)
    bestF = nowF
    runs += 1

    if abs(lastavg - currentavg) < difference:
      runs_in_range += 1
    elif abs(lastavg - currentavg) > difference and runs_in_range > 0:
      runs_in_range = 0
    
    if debug_flag:
      print("RUN #: ", runs)
      print("CURR AVG: ", currentavg)
      print("RUN IN RANGE #: ", runs_in_range)

  return bestF

def single_alpha_run(alpha, beta, strategy_sim, difference, times, debug_flag, lambda_end=0):
  if debug_flag: print("Alpha: ", alpha)

  # binary search for lambda
  l = 0
  start = 0
  end = lambda_end

  while (start <= end):
    l = (start + end)/2
    if debug_flag: print("Lambda: ", l)

    bestF = simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag)
    reward = np.average(bestF)

    if debug_flag: print("Reward for ", l, ": ", reward)
    
    # arbitrary error diff
    if abs(reward) < 0.01:
      break

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
      if abs(reward) < 0.01:
        break

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
  single_alpha_run(alpha=0.1, beta=0.5, strategy_sim=optplus_sim, difference=0.01, times=4, debug_flag=True, lambda_end=0.5)
  # gather_data(optplus_sim, beta=0.5, difference=0.01, times=4, output_file_name="OPTPLUS_beta=1_alpha=0.01-1.0.npy", debug_flag=True)
  
  # distributions = np.load("OPTPLUS distributions 0.01 to 1.0.npy")
  # gather_data_from_given_start(distributions, optplus_sim, start_alpha=0.86, 
  #   difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)


if __name__ == "__main__":
    main()

