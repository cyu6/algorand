# Code for Algorand Independent Work

This repository stores the code used to run simulations of an optimal strategy in Algorand.

## list of updates
* data can be found in "data" folder, "data/old" can be ignored, and the rest is separated by x=0 and x>0 runs, although the x>0 run data is almost certainly inaccurate
* "graphs" simply contains the interesting figures used in my JP and Thesis writeups
* "simulations" has the bulk of the interesting code. "run files" and "sanity checks" can be mostly ignored. the `constants.py` file stores the values of the global constants used in all simulation files. the `graphing_writeup.py` file primarily needs updating with the paths of the data files. the `simulation.py` file has the basic structure for running any simulations with x=0, while the `x_increase_run.py` file has the functions that run simulations with x>0.

## below this is out of date

The simulation code is mainly in the `simulation_final.py` file and the code used to create the graphs is in the `graphing_final.py` file. The writeup is also included in this repository.

The distributions produced by the simulation for the optimal cheating strategy and the COIN cheating strategy (see paper for more context) are saved as `npy` files and are essentially 100 x 10000 size arrays. The graphs can be found in the "graphs" file. 

*Update 2/24/2022*: after some discussion, we want to add to the defined OPT (optimal) cheating strategy by adding the case where accepting the honest network's coin as the seed for the next round results in higher expected rewards for the adversary than broadcasting any of the adversary's own coins. To simulate this more complex strategy (which we name OPTPLUS), we add the parameter $\lambda$:
* We want to answer the question "for given $\alpha$, is there a strategy that wins at least a $\lambda$ fraction of the rounds?".
* The difference for the OPTPLUS update rule compared to the OPT update rule is that for every round you win, you get (1-$\lambda$) points, and for every round you lose, you lose $\lambda$ points. 
* If you can win greater than or equal to $\lambda$ fraction of the rounds, you want to use that strategy.
* The code can be found in the `simulation_lambda.py` file and data from an initial run of the simulation is saved in a corresponding `npy` file.

To more easily view the data, I also included a file that basically prints the distribution data ($\alpha$ to $E[\alpha]$) for OPT and COIN (not yet updated for OPTPLUS, which also saves $\lambda$).
