#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 06:56:52 2017
https://pythonprogramming.net/how-to-program-r-squared-machine-learning-tutorial/?completed=/r-squared-coefficient-of-determination-machine-learning-tutorial/
@author: justin

calculate r^2 of tutorial 8 
"""


from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def best_fit_linear(xs, ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    b = mean(ys) - m * mean(xs)
    # coefficient determination using two different ways to calculate squared error
    r = 1 - np.sum((m*xs+b -ys)**2)/np.dot((ys-mean(ys)),(ys-mean(ys)))
    return m, b, r
    
xs = [1,2,3,4,5,6]
ys = [5,5,6,6,7,7]

xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

m,b,r = best_fit_linear(xs, ys)
print(m,b,r)

regression_line = m * xs + b

plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.show()
