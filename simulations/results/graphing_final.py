import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def graph_lambda_distribution(loaded_distributions, strategy, color, labels_flag):
  es = [k[0] for k in loaded_distributions]

  alphas = [*range(1, 101)]
  alphas = np.divide(alphas, 100)

  es = np.add(es, alphas)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  if labels_flag:
    for xy in zip(alphas, es):
      if (xy[0]*100) % 10 == 0:
        ax.annotate('(%s, %.4f)' % xy, xy=xy, textcoords='data') 
      
      # if xy[0] == 0.99:
      #   ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 

  ax.plot(alphas[:73], es[:73], color=color)
  ax.plot(alphas[:73], alphas[:73], color="blue")
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

  distributions = np.load("HASHPLUS_trial=1_beta=0.5_alpha=0.01-1.0.npy")

  display(distributions)

  graph_lambda_distribution(distributions, strategy="HASHPLUS", color="red", labels_flag=True)


if __name__ == "__main__":
    main()

