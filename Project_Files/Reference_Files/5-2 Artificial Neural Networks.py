# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:50:49 2023

@author: ehan1
"""

import pandas as pd
import numpy as np

## Import the data
# ALCHL_I: whether the driver is under the influence of alcohol or not
# PROFIL_I_R: whether the roadway is levelled or not
# SUR_COND: condition of the road surface (1: dry, 2: wet, 3: snow, 4: icy, 9: unknown)
# VEH_INVL: number of vehicles involved
# MAX_SEV_IR: severity of the accident (0: no injury, 1: non-fatal injury, 2: fatal injury)
accident_df = pd.read_csv("Accidents.csv")

## Dummifying predictors
accident_df["ALCHL_I"] = accident_df["ALCHL_I"]-1
accident_df = pd.get_dummies(accident_df, columns = ['SUR_COND','VEH_INVL'])

## Construct variables
X = accident_df.drop(columns=['MAX_SEV_IR'])
y = accident_df['MAX_SEV_IR']

## Split the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)

## Build a model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
model = mlp.fit(X_train,y_train)

## Make prediction and evaluate the performance
y_test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)

# Control the order of the label
confusion_matrix(y_test, y_test_pred, labels=[0,1,2])

# Print the label
print(pd.DataFrame(confusion_matrix(y_test, y_test_pred, labels=[0,1,2]), index=['true:0', 'true:1','true:2'], columns=['pred:0', 'pred:1','pred:2']))

## Varying the number of hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(11,11),max_iter=1000, random_state=0)
model2 = mlp2.fit(X_train,y_train)
y_test_pred_2 = model2.predict(X_test)
accuracy_score(y_test, y_test_pred_2)

## Cross-validate with different size of the hidden layer
from sklearn.model_selection import cross_val_score
for i in range (2,21):    
    model3 = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=0)
    scores = cross_val_score(estimator=model3, X=X, y=y, cv=5)
    print(i,':',np.average(scores))