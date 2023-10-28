#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:50:09 2023

@author: ehan1
"""

# Import Adult.csv and construct variables
import pandas as pd
import numpy as np

adult_df = pd.read_csv("Adult.csv")
adult_df = pd.get_dummies(adult_df, columns = ['race','sex','workclass','marital-status'])

# Construct variables
X = adult_df.iloc[:,1:]
y = adult_df.iloc[:,0]

# Split the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)


## 1. Classification tree
# Build a tree model with 3 layers
from sklearn.tree import DecisionTreeClassifier
ct = DecisionTreeClassifier(max_depth=3)  
model = ct.fit(X_train, y_train)

# Make prediction and evaluate accuracy
y_test_pred = model.predict(X_test)      

from sklearn.metrics import accuracy_score
accuracy_ct = accuracy_score(y_test, y_test_pred)      

# Print the tree
from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
features = X.columns
classes = ['<=50k','>50k']
tree.plot_tree(model,feature_names=features,class_names=classes,filled=True)
plt.show()

# Pruning pre-model building using K-fold cross validation for trees with different depths
from sklearn.model_selection import cross_val_score
for i in range (2,21):                                                 
    model = DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

# Pruning post-model building
ct = DecisionTreeClassifier()
path = ct.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_alphas = ccp_alphas[ccp_alphas >= 0]  

accuracy_ct = []
for ccp_alpha in ccp_alphas:
    ct = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    model = ct.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy_ct.append(accuracy_score(y_test, y_test_pred))

accuracy_ct = np.array(accuracy_ct)
alpha_accuracy = pd.DataFrame({'ccp_alpha': ccp_alphas, 'accuracy': accuracy_ct}, columns=['ccp_alpha', 'accuracy'])
accuracy_ccp = alpha_accuracy['accuracy'].max()
ccp_alpha = alpha_accuracy[alpha_accuracy['accuracy'] == accuracy_ccp]['ccp_alpha'].values[0]


## 2. Random Forest
# Build the model
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model_rf = randomforest.fit(X_train, y_train)

# Print feature importance
pd.Series(model_rf.feature_importances_,index = X.columns).sort_values(ascending = False).plot(kind = 'bar', figsize = (14,6))

# Make prediction and evaluate accuracy
y_test_pred = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_test_pred)

# K-fold cross validation for different numbers of features to consider at each split
for i in range (2,7):                                                                   
    model2 = RandomForestClassifier(random_state=0,max_features=i,n_estimators=100)
    scores = cross_val_score(estimator=model2, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
    
# Cross-validate internally using OOB observations
randomforest2 = RandomForestClassifier(random_state=0,oob_score=True)   
model3 = randomforest2.fit(X, y)
model3.oob_score_
# OOB scores only provide accuracy score
# If you want to look at other performance measures, you need to split the data into training and test sets instead of using OOB


## 3. Gradient Boosting Algorithm
# Build the model
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state=0)                           
model_gbt = gbt.fit(X_train, y_train)

# Make prediction and evaluate accuracy
y_test_pred = model_gbt.predict(X_test)
accuracy_gbt = accuracy_score(y_test, y_test_pred)

# K-fold cross-validation with different number of samples required to split
for i in range (2,10):                                                                        
    model2 = GradientBoostingClassifier(random_state=0,min_samples_split=i,n_estimators=100)
    scores = cross_val_score(estimator=model2, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

