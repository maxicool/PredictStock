# -*- coding: utf-8 -*-
"""
https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/?completed=/simple-linear-regression-machine-learning-tutorial/
"""

from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def best_fit_linear(xs, ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    b = mean(ys) - m * mean(xs)
    return m, b
    
xs = [1,2,3,4,5]
ys = [5,4,6,5,6]

xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

m,b = best_fit_linear(xs, ys)
print(m,b)

regression_line = m * xs + b

plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.show()
