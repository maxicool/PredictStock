#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 06:56:52 2017
https://pythonprogramming.net/how-to-program-r-squared-machine-learning-tutorial/?completed=/r-squared-coefficient-of-determination-machine-learning-tutorial/
@author: justin

random data generation
"""


from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import random

def best_fit_linear(xs, ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    b = mean(ys) - m * mean(xs)
    # coefficient determination using two different ways to calculate squared error
    r = 1 - np.sum((m*xs+b -ys)**2)/np.dot((ys-mean(ys)),(ys-mean(ys)))
    return m, b, r
    
def create_random_data(num, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(num):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        
        # add some correlation
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = np.arange(num)
    
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)


for variance in [40, 20, 10]:
    xs, ys =  create_random_data(100, variance, 2, 'neg')

    m,b,r = best_fit_linear(xs, ys)
    print(m,b,r)

    regression_line = m * xs + b

    plt.scatter(xs,ys,color='#003F72')
    plt.plot(xs, regression_line)
    plt.show()
