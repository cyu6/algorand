import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

n = 10000

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

def graph_lambda_distribution(loaded_distributions, strategy, color, labels_flag, plot_all):
  alphas = [*range(1, 101)]
  alphas = np.divide(alphas, 100)

  lambdas = [k[0] for k in loaded_distributions]
  lambdas = np.add(lambdas, alphas)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  if labels_flag:
    for xy in zip(alphas, lambdas):
      if (xy[0]*100) % 10 == 0:
        ax.annotate('(%s, %.4f)' % xy, xy=xy, textcoords='data') 
      
      # if xy[0] == 0.99:
      #   ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 

  ax.plot(alphas[:73], lambdas[:73], color=color)
  ax.plot(alphas[:73], alphas[:73], color="blue")

  if plot_all:
    opt_dist = np.load("data/OPT distributions 0.01 to 1.0.npy")
    coin_dist = np.load("data/COIN distributions 0.01 to 1.0.npy")
    opt_es = calculate_prop_es(opt_dist)
    coin_es = calculate_prop_es(coin_dist)
    ax.plot(alphas[:73], opt_es[:73], color="green")
    ax.plot(alphas[:73], coin_es[:73], color="orange")
    ax.legend([strategy, 'HONEST', 'OPT', 'COIN'])
  else:
    ax.legend([strategy, 'HONEST'])
  
  plt.xlabel("alpha")
  plt.ylabel("lambda")
  plt.show()

def display(loaded_distributions):
  es = [k[0] for k in loaded_distributions]
  for i in range(1, 100):
    alpha = i/100

    print(alpha)
    print(es[i-1])
  
  return es

def main():
  hashplus_dist = np.load("data/HASHPLUS_trial=1_beta=0.5_alpha=0.01-0.73.npy")

  graph_lambda_distribution(hashplus_dist, strategy="HASHPLUS", color="red", labels_flag=True, plot_all=False)


if __name__ == "__main__":
    main()

