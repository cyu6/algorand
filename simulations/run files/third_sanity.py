"""Strategies"""
from strategies import opt, coin, optplus, hashplus_exact, hashplus_sample
from hashplus_pooling import hashplus_sample_pool

from simulation import single_alpha_run

# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.5)
# LAST lambda was  0.0009765625

# single_alpha_run(alpha=0.01, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.07)


# single_alpha_run(alpha=0.1, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.5)
# single_alpha_run(alpha=0.1, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True, end_lambda=0.07)

####################################################################
# POST CODE EDITS

# single_alpha_run(alpha=0.3, beta=0.5, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True)
# 0.05468940734863281

# single_alpha_run(alpha=0.3, beta=0, strategy_sim=hashplus_sample, difference=0.01, times=4, debug_flag=True)
# 0.007812507479684427 --> 0.3078125075

# single_alpha_run(alpha=0.3, beta=0, strategy_sim=coin, difference=0.01, times=4, debug_flag=True, end_lambda=0)
# 0.4432849342162638 --> 0.3071361196

single_alpha_run(alpha=0.3, beta=0.5, strategy_sim=hashplus_sample_pool, difference=0.01, times=4, debug_flag=True)
# single_alpha_run(alpha=0.3, beta=0.5, strategy_sim=hashplus_sample_pool, difference=0.01, times=4, debug_flag=True)
# single_alpha_run(alpha=0.3, beta=0.5, strategy_sim=hashplus_sample_pool, difference=0.01, times=4, debug_flag=True)
# single_alpha_run(alpha=0.3, beta=0, strategy_sim=hashplus_sample_pool, difference=0.01, times=4, debug_flag=True, end_lambda=0.7)
# single_alpha_run(alpha=0.3, beta=0, strategy_sim=hashplus_sample_pool, difference=0.01, times=4, debug_flag=True, end_lambda=0.1)




# a=0.21, LAST lambda was  0.0283203125
# a=0.22, LAST lambda was  0.029296875
# a=0.23, LAST lambda was  0.0302734375
# a=0.24, LAST lambda was  0.03125
# a=0.25, LAST lambda was  0.03125
# a=0.26, LAST lambda was  0.0322265625
