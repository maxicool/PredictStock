# -*- coding: utf-8 -*-
"""
Spyder Editor

Tutorial 5
using pickle and scaling

https://pythonprogramming.net/forecasting-predicting-machine-learning-tutorial/?completed=/training-testing-machine-learning-tutorial/
"""
import pandas as pd
import quandl
import math, numpy as np
from sklearn import preprocessing, model_selection , svm  # cross_validation is deprecated, and is replaced to model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import datetime

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
percent_ahead = 1./100.                         # Try to predict percent_ahead of total dataframe
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
# using X = X[:-forecast_out] no need to drop na data right now, otherwise will shrink data
#df.dropna(inplace=True)

# Let X be input data (features), and y as output (labels)
# and convert to numpy array so that skelearn can handel the data
X = np.array(df.drop([label], 1))
# Standardize a dataset
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]       # features where y is [NaN]  
X = X[:-forecast_out]
y = np.array(df[label])
y = y[:-forecast_out]

# split data to tranning session and cross_validation session
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


# using LinearRegression from sklearn:
clf =  LinearRegression(n_jobs=-1)       # to open n_jobs threads (-1) using all possible threads
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('LinearRegression: %f' % confidence)

forecast_set = clf.predict(X_lately)      # using X_lately to predict forecast
print('I predict stock price in %d days' % forecast_out, forecast_set)

#iloc loc get local datetime ?
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()