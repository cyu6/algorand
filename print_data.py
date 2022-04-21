import math
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def display(loaded_distributions):
  es = np.zeros(100)
  for i in range(1, 100):
    alpha = i/100
    D = loaded_distributions[i-1,:]
    es[i] = np.average(D)

    print(alpha)
    print(es[i])
  
  return es

def display_lambda(distributions):
  for k in distributions:
    print(k[0])

def lambda_diff(dist, other):
  for i in range(100):
    print(dist[i][0] - other[i][0])

def main():

  # distributions = np.load("FINAL distributions 0.01 to 1.0.npy") # 10000, OPT, violet
  # distributions = np.load("COIN distributions 0.01 to 1.0.npy") # 10000, COIN, red

  # distributions = np.load("OPTPLUS distributions 0.01 to 1.0.npy") # 10000, COIN, red

  distributions = np.load("simulations/HASHPLUS_x=0.01_beta=0.5_alpha=0.01-1.0.npy")
  other = np.load("simulations/HASHPLUS_trial=2_beta=0.5_alpha=0.01-1.0.npy")

  display_lambda(distributions)
  print(np.count_nonzero(distributions))


if __name__ == "__main__":
    main()

