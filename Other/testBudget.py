import numpy as np
import math

start = 50000
deposit = 10000
interest = 1.05
interest_month = math.pow(1.05, 1/12.0)
num_months = 12*33
budget = np.zeros(num_months)
budget[0] = start
for k in range(1, num_months):
    budget[k] = budget[k-1]*interest_month + deposit

print(budget)
