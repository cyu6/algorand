import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from constants import i, n, m, x

###### Notes ######
# functions are correct
# paths for files need to be updated
# data files need to be standardized in format

# used for OPT and COIN
def calculate_prop_es(loaded_distributions):
  es = np.zeros(100)
  for i in range(1, 101):
    alpha = i/100
    es_alpha = alpha
    D = loaded_distributions[i-1,:]
    sorted_D = sorted(D)
    es_alpha += sorted_D[0]*alpha
    for j in range(1, n):
      index = j - 1
      interval = sorted_D[index+1] - sorted_D[index]
      prob = alpha - (((1-alpha)*alpha*j)/(n - alpha*j))
      es_alpha += (interval * prob)
    es[i-1] = es_alpha
  es = np.divide(es, np.add(1, es))
  return es

def lookahead(alpha):
  acc = 0
  for i in range(1, 1000):
      acc = acc + alpha**i*(1-alpha)*(1+(i*alpha)/(1+(i-1)*alpha))
  return acc/(1+alpha)

def upperbound(alpha):
  return alpha*(2-alpha)/(1-alpha)

def algorandbound(alpha):
  return 1 - ((1-alpha)**2)*(1+alpha-alpha**2)

def graph_distributions(strategy, labels_flag, plot_all):
  alphas = [*range(1, 101)]
  alphas = np.divide(alphas, 100)

  bquarter = np.add(np.load("beta=0.25_alpha=0.01-0.64.npy"), alphas[:66])
  bhalf = np.load("beta=0.50_alpha=0.01-0.68_hannah.npy")
  bthreequarter = np.add(np.load("beta=0.75_alpha=0.01-0.20.npy"), alphas[:20])
  bone_cat = np.add(np.load("beta=1_alpha=0.01-0.54_cat.npy"), alphas[:56])
  bone_hannah = np.load("beta=1_alpha=0.01-0.60_hannah.npy")
  # xrandom = np.add(np.load("x=0.01_beta=0.5_alpha=0.01-0.32.npy"), alphas[:32])

  upper_bound = [upperbound(a) for a in alphas]
  onelook = [lookahead(a) for a in alphas]
  algorand_bound = [algorandbound(a) for a in alphas]

  fig = plt.figure()
  ax = fig.add_subplot(111)

  if labels_flag:
    for xy in zip(alphas, bhalf):
      if (xy[0]*100) % 5 == 0:
        ax.annotate('(%s, %.4f)' % xy, xy=xy, textcoords='data') 
    
    # for xy in zip(alphas, bthreequarter):
    #   if (xy[0]*100) % 5 == 0:
    #     ax.annotate('(%s, %.4f)' % xy, xy=xy, textcoords='data') 
      
      # if xy[0] == 0.99:
      #   ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 

  slice_index = 30

  # ax.plot(alphas[:slice_index], xrandom[:slice_index], color="darkturquoise")
  ax.plot(alphas[:slice_index], bone_hannah[:slice_index], color="black")
  # ax.plot(alphas[:slice_index], bone_cat[:slice_index], color="deeppink")
  # ax.plot(alphas[:slice_index], bthreequarter[:slice_index], color="purple")
  ax.plot(alphas[:slice_index], bhalf[:slice_index], color="red")
  ax.plot(alphas[:slice_index], bquarter[:slice_index], color="teal")
  # ax.plot(alphas[:slice_index], upper_bound[:slice_index], color="deeppink")
  # ax.plot(alphas[:slice_index], algorand_bound[:slice_index], color="aquamarine")
  # ax.plot(alphas[:slice_index], onelook[:slice_index], color="sienna")
  
  if plot_all:
    opt_dist = np.load("../data/OPT distributions 0.01 to 1.0.npy")
    coin_dist = np.load("../data/COIN distributions 0.01 to 1.0.npy")
    opt_es = calculate_prop_es(opt_dist)
    coin_es = calculate_prop_es(coin_dist)
    ax.plot(alphas[:slice_index], opt_es[:slice_index], color="darkviolet")
    ax.plot(alphas[:slice_index], coin_es[:slice_index], color="orange")
    ax.plot(alphas[:slice_index], alphas[:slice_index], color="blue")
    ax.legend([strategy+r"$(\beta=1)$", strategy+r"$(\beta=0.50)$", strategy+r"$(\beta=0.25)$", 'OPT', 'COIN', 'HONEST'])
  else:
    ax.plot(alphas[:slice_index], alphas[:slice_index], color="blue")
    # ax.legend([strategy+r"$(\beta=1)$", "Algorand Bound", "1-Lookahead", "HONEST"])
    ax.legend([strategy+r"$(\beta=1)$", strategy+r"$(\beta=0.75)$", strategy+r"$(\beta=0.50)$", strategy+r"$(\beta=0.25)$", 'HONEST'])
  
  plt.xlabel(r'$\alpha$')
  plt.ylabel(r'Revenue($\pi$)')
  # plt.title(r"Revenue for different $\beta$ values")
  plt.title(r'LOCAL-LOSS vs OPT vs COIN vs HONEST')
  plt.show()

def display(loaded_distributions):
  es = [k[0] for k in loaded_distributions]
  for i in range(1, 100):
    alpha = i/100

    print(alpha)
    print(es[i-1])
  
  return es

def samplediff():
  alphas = [*range(1, 101)]
  alphas = np.divide(alphas, 100)
  bone_cat = np.add(np.load("beta=1_alpha=0.01-0.54_cat.npy"), alphas[:56])
  bone_hannah = np.load("beta=1_alpha=0.01-0.60_hannah.npy")
  mse = 0
  for i in range(7, 56):
    mse += abs(bone_cat[i] - bone_hannah[i])**2
  mse /= 49
  print(mse)

def plotx(xs, lambdas):
  alphas = [*range(1, 101)]
  alphas = np.divide(alphas, 100)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  slice_index = 0

  # bquarter = np.load("beta=0.25_alpha=0.01-0.64.npy")
  # bhalf = np.subtract(np.load("beta=0.50_alpha=0.01-0.68_hannah.npy"), alphas)
  bone = np.load("beta=1_alpha=0.01-0.54_cat.npy")
  # 0 index -> 0.01 alpha
  # 24 index -> 0.25 alpha

  ax.plot(xs, lambdas, color="blue")
  
  for xy in zip(xs, lambdas):
    if (xy[0] in [0.0, 0.5, 1.0, 2.0, 4.0, 5.0]):
      ax.annotate('%.4f' % xy[1], xy=xy, textcoords='data') 
  
  ax.plot(xs[:], [bone[24]]*47, color="black")
  # ax.plot(xs[:], [bhalf[24]]*47, color="green")
  # ax.plot(xs[:], [bquarter[24]]*47, color="purple")
  ax.legend([r"LOCAL-LOSS($\beta=1, x$)", r"LOCAL-LOSS($\beta=1, x=0$)"])
  
  plt.xlabel(r'$x$')
  plt.ylabel(r'$\lambda$')
  # plt.title(r"Revenue for different $\beta$ values")
  plt.title(r'Fraction of rounds won over $x$ for LOCAL-LOSS($\alpha=0.25, \beta=1$)')

  plt.show()

def main():
  # hashplus_dist = np.load("data/HASHPLUS_trial=1_beta=0.5_alpha=0.01-0.73.npy")

  # samplediff()
  graph_distributions(strategy="LOCAL-LOSS", labels_flag=False, plot_all=True)

  previous_x = np.load("X_INCREASE_0.1-10_alpha=0.25_beta=1.npy")
  prev_xs = [k[0] for k in previous_x][:41]
  prev_ls = [k[1] for k in previous_x][:41]
  print(prev_xs)
  print(prev_ls)
  # xstuff = np.load("x_increase_alpha=0.25_beta=1.npy")
  xstuff = np.load("x_increase_alpha=0.25_beta=1.npy")[-6:]
  xs = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
  print(xs)
  print(xstuff)
  prev_xs.extend(xs)
  prev_ls.extend(xstuff)
  # plotx(prev_xs, prev_ls)
  



if __name__ == "__main__":
    main()

