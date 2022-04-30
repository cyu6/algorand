from simulation import gather_data_from_given_start, simulate
from hashplus_pooling import hashplus_sample_pool
import numpy as np

def one_alpha_run(distributions, num, beta, difference, times, output_file_name, debug_flag):
  alpha = num/100

  if debug_flag: print("ALPHA: ", alpha)

  # binary search for lambda
  l = 0  # same as mid
  start = 0
  end = 1

  while (start <= end):
    l = (start + end)/2
    if debug_flag: print("Lambda: ", l)

    bestF = simulate(alpha, beta, l, hashplus_sample_pool, difference, times, debug_flag)
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

# distributions = np.load("HASHPLUS_trial=2_beta=0.5_alpha=0.01-1.0.npy")
from constants import n

distributions = np.zeros((100, n+1))
# distributions = np.load("/content/gdrive/MyDrive/HASHPLUS_beta=0.75_alpha=0.01-1.0.npy")

# alpha=0.50
beta=1
difference=0.01
times=4
output_file_name="HASHPLUS_makeup_beta=1_alpha=0.01-0.06.npy"
debug_flag=True

# distributions = np.load("HASHPLUS_trial=2_beta=0.5_alpha=0.01-1.0.npy")

one_alpha_run(distributions, 6, beta, difference, times, output_file_name, debug_flag)

# 0.005859375, 0.01171875, 0.01953125, 0.02734375, 0.033203125, 0.0419921875