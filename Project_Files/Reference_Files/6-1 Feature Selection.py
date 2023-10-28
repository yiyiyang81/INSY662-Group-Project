#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:47:40 2023

@author: ehan1
"""

## Import data and construct variables (use all predictors)
import pandas as pd
import numpy as np

bank_df = pd.read_csv("UniversalBank.csv")

X = bank_df.iloc[:,2:13]
y = bank_df["Personal Loan"]


##### Build ANN model without feature selection #####
## Split the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)

## Build ANN with one hidden layer and 11 nodes
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
model = mlp.fit(X_train,y_train)

## Make prediction and evaluate accuracy
y_test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)

## Varying the number of hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(11,11),max_iter=1000, random_state=0)
model2 = mlp2.fit(X_train,y_train)
y_test_pred_2 = model2.predict(X_test)
accuracy_score(y_test, y_test_pred_2)

## Cross-validate with different size of the hidden layer
from sklearn.model_selection import cross_val_score
for i in range (7,16):    
    model3 = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=0)
    scores = cross_val_score(estimator=model3, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

## ANN with optimal size of hidden layer from above
mlp = MLPClassifier(hidden_layer_sizes=(7),max_iter=1000, random_state=0)
model = mlp.fit(X_train,y_train)
y_test_pred = model.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_test_pred)


##### Apply feature selection ##### 
## 1. LASSO
# Standardize predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Run LASSO with alpha=0.01
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01) # you can control the number of predictors through alpha
model = ls.fit(X_std,y)

pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])

# Fit the model on new dataset and evaluate
X_lasso_train = X_train.drop(['Age','Experience','Mortgage','Online'], axis=1)
X_lasso_test = X_test.drop(['Age','Experience','Mortgage','Online'], axis=1)

model = mlp.fit(X_lasso_train,y_train)
y_test_pred = model.predict(X_lasso_test)
accuracy_ann_lasso = accuracy_score(y_test, y_test_pred)


## 2. Random forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X, y)

pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])
# Can specify a threshold for the feature importance; common is 0.05 but it's always subjective

# Fit the model on new dataset and evaluate
X_rf_train = X_train[['Income','Family','CCAvg','Education']]
X_rf_test = X_test[['Income','Family','CCAvg','Education']]

model = mlp.fit(X_rf_train,y_train)
y_test_pred = model.predict(X_rf_test)
accuracy_ann_rf = accuracy_score(y_test, y_test_pred)


## 3. Principal component analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=11)
pca.fit(X_std)

pca.explained_variance_ratio_

pca.components_

# Using elbow method to select number of components
import matplotlib.pyplot as plt
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Fit the model on new dataset and evaluate
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_std)
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.33, random_state = 5)

model = mlp.fit(X_pca_train,y_train)
y_test_pred = model.predict(X_pca_test)
accuracy_ann_pca = accuracy_score(y_test, y_test_pred)


## 4. Recursive feature elimination (Doesn't work with MLPclassifier, thus using logistic regression for demonstration) 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=5000)
rfe = RFE(lr, n_features_to_select=3) #need to specify how many features you want at the end
model = rfe.fit(X, y)
model.support_ #shows whether each feature is used or not as a boolean value

pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking']) #ranks features by their importance
