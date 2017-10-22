#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:02:12 2017
Creating a K Nearest Neighbors Classifer from scratch


https://pythonprogramming.net/programming-k-nearest-neighbors-machine-learning-tutorial/?completed=/euclidean-distance-machine-learning-tutorial/
@author: justin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
#dont forget this
import pandas as pd
import random
style.use('fivethirtyeight')

def k_nearest_neighbors(data, to_predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:              
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(to_predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)
    
    #print (vote_result)
    return vote_result[0][0]

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.fillna(-99999, inplace = True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

#print (full_data[:3])
random.shuffle(full_data)
#print(full_data[:3])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)

import pickle
pickle_file = 'KNeighborsCLF'

pickle_in = open(pickle_file,'rb')
clf = pickle.load(pickle_in)
for group in test_set:
    for data in test_set[group]:
        data = np.array([data])
        #data = data[np.newaxis]
        data.reshape(1,-1)
        #print(data)
        #print(data.shape)
        #print(len(data))
        vote = clf.predict(data)[0]
        #print(vote)
        
        
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)

