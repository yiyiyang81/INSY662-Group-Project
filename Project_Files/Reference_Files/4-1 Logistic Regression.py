# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:18:29 2023

@author: ehan1
"""
## Import data and construct variables
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("UniversalBank.csv")

# Dummify variables
df = pd.get_dummies(df, columns = ['Education'], drop_first=True)

# Construct variables
X = df.iloc[:,2:14]
y = df["Personal Loan"]


## 1. using statsmodels to run logistic regression
import statsmodels.api

X1 = statsmodels.api.add_constant(X)
logit = statsmodels.api.Logit(y,X1)
model = logit.fit()
model.summary()


## 2. using sklearn to run logistic regression
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

# Run the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression() # automatically runs a regularized logistic regression; to control the penalty, add argument 'C = value'; C represents the inverse of lambda
model = lr.fit(X_train,y_train) 

# can set higher 'max_iter' if the warning message appears
lr = LogisticRegression(max_iter=5000)
model = lr.fit(X_train,y_train)

# View results
model.intercept_
model.coef_

# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)

# Using the model to predict the probability of being classified to each category
y_test_pred_prob = model.predict_proba(X_test)[:,1]


## 3. Get performance measures
# Accuracy score
from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)

# Confusion matrix
metrics.confusion_matrix(y_test, y_test_pred)

# Confusion matrix with label
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1']))

# Precision/Recall
metrics.precision_score(y_test, y_test_pred)
metrics.recall_score(y_test, y_test_pred)

# F1 score
metrics.f1_score(y_test, y_test_pred)

# ROC curve and AUC score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_test_pred_prob1 = y_test_pred_prob[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob1)
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

roc_auc_score(y_test, y_test_pred_prob)