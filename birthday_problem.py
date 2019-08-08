# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:31:30 2019

@author: dmitri4
"""

## Monte Carlo simulation --> Birthday Problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
The birthday problem, or rather the birthday paradox, is the relatively small
number of people needed in a crowd for any two people to share a birthday.

https://en.wikipedia.org/wiki/Birthday_problem

Solutions using empirical probabilities generally converge on 23 people needed
for a 50% chance of a shared birthday between any two individuals. At first glance,
this is a surprisingly small number considering 366 possible birthdays.

However, odds quickly increase since we're covering 22 other people every time
we consider a given individual's possible birthday matches. This can be simulated
in Python using Monte Carlo techniques.

The following is my code from a lab assignment in Systems Science 535 - Modeling &
Simulation with R & Python (PSU, Spring 2019).

-Dmitri Kalashnikov
'''

# defining the function
# arguments are the number of people in the crowd, and the number of Monte Carlo sims to perform

def birthday_odds(num_people, num_sims):
    results = np.empty([num_sims])
    for k in range(num_sims):
        crowd = np.empty([num_people])
        for i in range(num_people):
            person = np.random.randint(1,366) # 366 possible birthdays
            crowd[i] = person
        dups = pd.Series(crowd)[pd.Series(crowd).duplicated()].values # searching for dups --> 2 birthdays match
        size = np.size(dups) # will have size = 0 unless two birthdays matched
        if size > 0:
            results[k] = 1 # boolean to indicate 2 birthdays matched
        else:
            results[k] = 0
    matches = np.size(results[results == 1])
    result = (matches/num_sims) * 100 # odds expressed as percentage
    return result

# iterating for crowd sizes from 2 to 30 people

probabilities = np.empty([29])
for j in np.arange(2,31,1):
    odds = birthday_odds(j,1000) # choosing 1000 simulations
    probabilities[j - 2] = odds

# plotting results

plt.plot(probabilities, drawstyle = 'steps')
plt.xticks(np.arange(5,35,5))
plt.xlabel('Number of people (n)')
plt.ylabel('P(n) - expressed as %')
plt.title('Probabilities of two people sharing birthday')
plt.show()

# printing results for input to table

print(probabilities)
