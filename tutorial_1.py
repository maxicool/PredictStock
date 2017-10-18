# -*- coding: utf-8 -*-
"""
Spyder Editor

Tutorial 1
data frame and quandl

https://pythonprogramming.net/features-labels-machine-learning-tutorial/?completed=/regression-introduction-machine-learning-tutorial/

"""
import pandas as pd
import quandl
import math, numpy as np
from sklearn import preprocessing, model_selection , svm  # cross_validation is deprecated, and is replaced to model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
"""
API access key quandl:
AgCio8CTB7ydqxaRtz4D
"""

quandl.ApiConfig.api_key = 'AgCio8CTB7ydqxaRtz4D'
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open',  'Adj. High',   'Adj. Low',  'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume',]]
print(df.head(3))
