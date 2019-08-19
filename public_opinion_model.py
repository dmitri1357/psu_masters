# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:27:46 2019

@author: dmitri4
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

'''
The Zaller-Deffuant Model of Mass Opinion (https://arxiv.org/abs/0908.2519) is an
Agent-based model (ABM) that attemps to show how public opinion converges over time.
Convergence occurs after many pairwise one-on-one interactions among the population,
leading individuals to move closer in their opinions regarding a subject matter.

The following is my code from a lab assignment in Systems Science 535 - Modeling &
Simulation with R & Python (PSU, Spring 2019).

1) My program implements the Zaller-Deffuant model using suggested convergence and
threshold parameters from published literature. An initial population of opinions
was drawn from a Laplace distribution, with population size of 20,000
(following Sobkowicz 2015) and lambda of 0.1. The decay parameter was chosen
empirically as a compromise between central tendency of opinions and outer
boundaries near 1. The few values outside of -1 to 1 interval were manually
set to 1 or -1, respectively. An initial kurtosis statistic of this population
was calculated in order to quantify the "peakedness" of this distribution.

2) Following, 10,000 pairwise exchanges of opinions were simulated, with
opinions i & j randomly drawn from the Laplace distribution. I chose to do this
sampling with replacement, in order to simulate real-life interactions where
each person has the freedom to talk to many other people. A new sample of
20,000 opinions was thus built, and the new kurtosis statistic calculated
to see if distribution became more "peaked" or platykurtic - a sign of
converging opinions.

3) A Monte Carlo simulation of above procedure was performed n = 1,000 times.

-Dmitri Kalashnikov
'''
u = 0.2 # convergence rate
d = 0.6 # global threshold
kurt_init = np.empty(1000)
kurt_after = np.empty(1000)
for i in range(1000): # population-wide exchange of opinions simulated 1000 times
    population = np.random.laplace(loc = 0, scale = 0.1, size = 20000) # population of 20,000 individual opinions
    population[population > 1] = 1 # set upper boundary at 1
    population[population < -1] = -1 # set lower boundary at -1
    kurt_init[i] = stats.kurtosis(population) # initial kurtosis of population, before exchange of opinions
    result_i = np.empty(10000)
    result_j = np.empty(10000)
    for k in range(10000): # 10,000 pair-wise interactions leading to opinion exchanges
        oi_t = np.random.choice(population) # initial opinion of person i, randomly drawn from population
        oj_t = np.random.choice(population) # initial opinion of person j, randomly drawn from population
        if np.abs(oi_t - oj_t) <= d: # threshold check
            oi_t1 = oi_t + u * (oj_t - oi_t) # opinion of person i changes
            oj_t1 = oj_t + u * (oi_t - oj_t) # opinion of person j changes
            result_i[k] = oi_t1 # new opinion of person i
            result_j[k] = oj_t1 # new opinion of person j
        else:
            result_i[k] = oi_t # opinion of person i stayed the same
            result_j[k] = oj_t # opinion of person j stayed the same
    results = np.hstack([result_i, result_j]) # new population of 20,000 opinions, from 10,000 interactions
    kurt_after[i] = stats.kurtosis(results) # kurtosis of population after exchange of opinions

# The initial kurtosis statistic indicates near-normal distribution,
# as 3.0 is considered normal or Gaussian.
plt.plot(kurt_init)
plt.title('Initial kurtosis statistics for 1000 simulated populations')
plt.xlabel('nsim = 1000')
plt.ylabel('Kurtosis')
plt.show()

print('Average initial kurtosis statistic:')
np.mean(kurt_init)

# A value of 4.24 indicates a more platykurtic distribution, as opinion values
# have converged toward the center of the distibution.
plt.plot(kurt_after)
plt.title('Kurtosis after exchange of opinions')
plt.xlabel('nsim = 1000')
plt.ylabel('Kurtosis')
plt.show()

print('Average kurtosis after 10,000 opinion exchanges:')
np.mean(kurt_after)

# On average, kurtosis increased by ~ 1.3 showing that this model consistenly
# predicts a convergence of opinions after 10,000 interactions.
kurt_diff = kurt_after - kurt_init

plt.plot(kurt_diff)
plt.title('Increase in kurtosis after opinion exchanges')
plt.xlabel('nsim = 1000')
plt.ylabel('Kurtosis increase')
plt.show()

print('Average increase in kurtosis after 10,000 opinion exchanges:')
np.mean(kurt_diff)









