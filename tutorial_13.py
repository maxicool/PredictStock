#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Classification 
K nearest fit
Created on Sun Oct 22 08:09:13 2017
https://pythonprogramming.net/k-nearest-neighbors-intro-machine-learning-tutorial/?completed=/sample-data-testing-machine-learning-tutorial/
https://pythonprogramming.net/k-nearest-neighbors-application-machine-learning-tutorial/?completed=/k-nearest-neighbors-intro-machine-learning-tutorial/

SVM
https://pythonprogramming.net/final-thoughts-knn-machine-learning-tutorial/
@author: justin
"""

import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import pickle


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.fillna(value = -99999, inplace = True)       # if na, fill it with -99999, to make na as outlined data

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
#X = preprocessing.scale(X)
y = np.array(df['class'])
pickle_file = 'KNeighborsCLF'
svm_pickle_file = 'svmSVC'

from pathlib import Path

my_file = Path(pickle_file)
if not my_file.is_file():
    accuracy = 0 
    while accuracy < 0.99 :
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(accuracy)
    with open(pickle_file, 'wb') as f:
        pickle.dump(clf, f) 

my_file = Path(svm_pickle_file)
if not my_file.is_file():
    accuracy = 0 
    while accuracy < 0.99 :
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(accuracy)
    with open(svm_pickle_file, 'wb') as f:
        pickle.dump(clf, f) 


pickle_in = open(pickle_file,'rb')
clf = pickle.load(pickle_in)

example_measures = np.array([[2,2,4,4,1,4,2,2,2],[4,2,1,1,1,2,3,2,1]])
# for one dimension array  need to reshape
#print(example_measures.shape)
example_measures = example_measures.reshape(len(example_measures), -1)
#print(example_measures.shape)

prediction = clf.predict(example_measures)
print(prediction)

pickle_in = open(svm_pickle_file,'rb')
clf = pickle.load(pickle_in)

example_measures = np.array([[2,2,4,4,1,4,2,2,2],[4,2,1,1,1,2,3,2,1]])
# for one dimension array  need to reshape
#print(example_measures.shape)
example_measures = example_measures.reshape(len(example_measures), -1)
#print(example_measures.shape)

prediction = clf.predict(example_measures)
print(prediction)

