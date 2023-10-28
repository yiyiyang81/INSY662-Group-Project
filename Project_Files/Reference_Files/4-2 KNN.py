# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:30:39 2023

@author: ehan1
"""

## 1. Illustrative example from the slides
# Load libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# Create a dataframe
data = np.array([['drug_z',0.0467,0.2471],['drug_y',0.0533,0.1912],['drug_x',0.0917,0.2794]]) # already normalized
column_names = ['Drug', 'Age (MMN)', 'Na/K (MMN)']
row_names  = ['A', 'B', 'C']
df = pd.DataFrame(data, columns=column_names, index=row_names)

# Construct variables
X = df.iloc[:,1:3]
y = df['Drug']

# Build a model
knn = KNeighborsClassifier(n_neighbors=1)
model = knn.fit(X,y)

# Make prediction for a new observation (age = 0.05, Na/K = 0.25)
new_obs = [[0.05,0.25]]
model.predict(new_obs)


## 2. Build KNN model using cancer.csv
# Load data and construct variables
df = pd.read_csv("cancer.csv")
X = df.iloc[:,0:9]
y = df['class']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

# Standardize the dataset
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
scaled_X_train = standardizer.fit_transform(X_train)
scaled_X_test = standardizer.transform(X_test)

# Build a model with k = 3 and using euclidean distance function
knn = KNeighborsClassifier(n_neighbors=3,p=2)
model = knn.fit(scaled_X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(scaled_X_test)

# Get accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)


## 3. Choosing k
for i in range (15,25):
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(scaled_X_train,y_train)
    y_test_pred = model.predict(scaled_X_test)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))
    

## 4. Make prediction for a new observation using optimal K
new_obs = [[4,2,1,1,1,8,3,1,1]]
scaled_new_obs = standardizer.transform(new_obs)

knn = KNeighborsClassifier(n_neighbors=19)
model = knn.fit(scaled_X_train,y_train)
model.predict(scaled_new_obs)