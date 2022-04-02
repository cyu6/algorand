"""Strategies"""
import opt, coin, optplus, hashplus_exact, hashplus_sample

from simulation import single_alpha_run

# single_alpha_run(alpha=0.1, beta=0, strategy_sim=hashplus_exact.hashplus_exact, difference=0.01, times=4, debug_flag=True, end_lambda=0)
# LAST lambda was 0.00390625

# single_alpha_run(alpha=0.1, beta=1, strategy_sim=hashplus_sample.hashplus_sample, difference=0.01, times=4, debug_flag=True)
# for beta = 0:
  # LAST lambda was 0.046875 + alpha ~~ 0.1046875
  # after the change, it became 0.0078125 :(((
# for beta = 1:
  # LAST lambda was 0.0625

# single_alpha_run(alpha=0.1, beta=0, strategy_sim=coin.coin, difference=0.01, times=4, debug_flag=True, end_lambda=0)
# reward approx 0.11300767547886553
# fraction = 0.113/(1+0.113) ~~ 0.105

# single_alpha_run(alpha=0.1, beta=1, strategy_sim=opt.opt, difference=0.01, times=4, debug_flag=True, end_lambda=0)
# reward approx 0.11494220523904715
# fraction = 0.115(1+0.115) ~~ 0.103


##########

# alpha = 0.3

single_alpha_run(alpha=0.3, beta=0, strategy_sim=hashplus_sample.hashplus_sample, difference=0.01, times=4, debug_flag=True)
# for beta = 0:
# LAST lambda was 0.0078125 + alpha = 0.30078125
# same for second trial

single_alpha_run(alpha=0.3, beta=1, strategy_sim=hashplus_sample.hashplus_sample, difference=0.01, times=4, debug_flag=True)
# for beta = 1: 
# Last lambda was 0.109375 + alpha = 0.409375
# second trial last lambda was 0.1015625 + alpha = 0.4015625

single_alpha_run(alpha=0.3, beta=0, strategy_sim=coin.coin, difference=0.01, times=4, debug_flag=True, end_lambda=0)
# reward approx 0.4479436989522651
# fraction = 0.448/(1.448) = 0.309
# second trial reward 0.43572233869911325
# fraction = 0.436/1.436 = 0.303

single_alpha_run(alpha=0.3, beta=1, strategy_sim=opt.opt, difference=0.01, times=4, debug_flag=True, end_lambda=0)
# reward approx 0.4773451116326473
# fraction = 0.477/(1.477) = 0.323
# second trial reward 0.47166498101223414
# fraction = 0.472/1.472 = 0.320

################
# Things of note

# hashplus beta=0 seems to do worse than COIN, not sure why
# checked and the reward each round is calculated the same, so I do feel like the lambda alpha subtraction
# per round IS getting included in the recursion and affecting the overall rewards (i.e. adding alpha at the end isn't enough to make up for it)
# want to confirm that for beta=0, we should have k=n and r_pdf = 0

# i guess it's not out of the ordinary for hashplus beta=1 to be a lot better than OPT but idk it does seem like a lot
