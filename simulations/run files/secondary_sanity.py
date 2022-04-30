"""Strategies"""
from strategies import opt, coin, optplus, hashplus_exact, hashplus_sample

from simulation import single_alpha_run

################
# different lambda starts

# lambda = 0.5 start
# single_alpha_run(alpha=0.3, beta=1, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True)
# end lambda = 0.1015625

# lambda = 0.25 start
# single_alpha_run(alpha=0.3, beta=1, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.5)
# end lambda = 0.1064453125
# the reward @ 0.1015625 was really close though, 0.0012 ish

# lambda = 0.125 start
# single_alpha_run(alpha=0.3, beta=1, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.25)
# interesting -- reward for 0.1015625 :  0.02414205500693666
# LAST lambda was  0.105712890625

# lambda = 0.07 start
# single_alpha_run(alpha=0.3, beta=1, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.14)
# LAST lambda was  0.10500000000000001

# lambda = 0.05 start
# single_alpha_run(alpha=0.3, beta=1, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.1)
# binary search stopped at 0.1 (wouldn't go above the end)


# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True)
# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.5)
# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.25)
# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.125)

# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.015625)
# Reward for  0.00048828125 :  -3.200852449970686e-05
# LAST lambda was  0.00048828125


