# -*- coding: utf-8 -*-
"""
Spyder Editor

Tutorial 3
Regression - Training and Testing

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

# add forecast_col to dataframe
forecast_col = 'Adj. Close'
df.fillna(value = -99999, inplace = True)       # if na, fill it with -99999, to make na as outlined data
percent_ahead = 1./1000                         # Try to predict percent_ahead of total dataframe
forecast_out = int(math.ceil(percent_ahead * len(df))) 

label = 'Price in ' + str(forecast_out)
df[label] = df[forecast_col].shift(-forecast_out)   # add forecast column and set values

"""
print (df.head())
print (df.tail())

DataFrame kind of like:
    quandl dataframe always keeps Date ?
             Adj. Close    HL_PCT  PCT_change  Adj. Volume  Price in 7
Date                                 
2004-08-19   50.322842  8.441017    0.324968   44659000.0   51.162935
2004-08-20   54.322689  8.537313    7.227007   22834300.0   51.343492
2004-08-23   54.869377  4.062357   -1.227880   18256100.0   50.280210
.....................................................................   
2017-10-13     1007.87  0.764602   -0.122881    1308881.0         NaN
2017-10-16     1009.35  1.046409   -0.027733    1066744.0         NaN
2017-10-17     1011.00  0.845882    0.353371     991412.0         NaN
"""
# drop last NaNs so that learning can be processed
df.dropna(inplace=True)

# Let X be input data (features), and y as output (labels)
# and convert to numpy array so that skelearn can handel the data
X = np.array(df.drop([label], 1))
X = preprocessing.scale(X)
y = np.array(df[label])

# split data to tranning session and cross_validation session
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)


# using LinearRegression from sklearn:
clf =  LinearRegression(n_jobs=-1)       # to open n_jobs threads (-1) using all possible threads
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('LinearRegression: %f' % confidence)

#to use Support Vector Regression from Scikit-Learn's svm package:
for k in ['linear','poly','rbf','sigmoid']:
    print(k)
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
