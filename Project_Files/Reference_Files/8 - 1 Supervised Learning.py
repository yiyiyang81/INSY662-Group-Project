#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:09:42 2023

@author: ehan1
"""

# Importing Pandas and NumPy
import pandas as pd, numpy as np, seaborn as sns,matplotlib.pyplot as plt

df = pd.read_csv('attrition.csv')

df.shape

df.info()

pd.crosstab(index = df['Attrition'], columns = 'counts')



# some missing values detected!

### Missing values
# 1. EnvironmentSatisfaction
sns.countplot(x='EnvironmentSatisfaction',data=df) # plot the counts in a boxplot

# The median value is 'high'. Thus, replace NAs with 'High'
df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].fillna('High')

# 2. JobSatisfaction
sns.countplot(x='JobSatisfaction',data=df) # plot the counts in a boxplot

# The median value is 'high'. Thus, replace NAs with 'High'
df['JobSatisfaction'] = df['JobSatisfaction'].fillna('High')

# 3. WorkLifeBalance
sns.countplot(x='WorkLifeBalance',data=df) # plot the counts in a boxplot

# The median value is 'better'. Thus, replace NAs with 'better'
df['WorkLifeBalance'] = df['WorkLifeBalance'].fillna('Better')

# 4. NumCompaniesWorked
df['NumCompaniesWorked'].describe()

# The median value is 2. Thus, replace NAs with 2
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(2)

# 4. TotalWorkingYears 
df['TotalWorkingYears'].describe()

# The median value is 10. Thus, replace NAs with 10
df['TotalWorkingYears'] = df['TotalWorkingYears'].fillna(10)



### EDA for insights
# 1. Work life balance and attrition
plt.figure(figsize=(8,8))
ax = sns.countplot(x='WorkLifeBalance',data=df,hue="Attrition") 
groups = df['WorkLifeBalance'].unique() # get categories of a variable
proportions = df.groupby('WorkLifeBalance')['Attrition'].value_counts(normalize=True) # get percentage for each category
hue_type = df['Attrition'].dtype.type
#for loop to add percentage label to each bar
for c in ax.containers:
    labels = [f'{proportions.loc[g, hue_type(c.get_label())]:.1%}' for g in groups]
    ax.bar_label(c, labels)

# 2. Business travel and attrition
plt.figure(figsize=(8,8))
ax = sns.countplot(x='BusinessTravel',data=df,hue="Attrition") 
groups = df['BusinessTravel'].unique() # get categories of a variable
proportions = df.groupby('BusinessTravel')['Attrition'].value_counts(normalize=True) # get percentage for each category
hue_type = df['Attrition'].dtype.type
#for loop to add percentage label to each bar
for c in ax.containers:
    labels = [f'{proportions.loc[g, hue_type(c.get_label())]:.1%}' for g in groups]
    ax.bar_label(c, labels)
    
# 3. Total working years and attrition
plt.figure(figsize=(8,8))
sns.boxplot(x="Attrition", y="TotalWorkingYears", data=df)

# you can do many more!


### Pre-processing
# 1. Drop irrelevant variables
# EmployeeID is just an identifier
# Over18 is a unary variable
df = df.drop(['EmployeeID', 'Over18'], axis=1)

# 2. Dummify
# Creating a dummy variable for some of the categorical variables and dropping the first one
dummy = pd.get_dummies(df[['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
                                 'JobSatisfaction', 'WorkLifeBalance','BusinessTravel', 'Department',
                                 'Education','EducationField', 'Gender', 'JobLevel', 'JobRole',
                                 'MaritalStatus']], drop_first=True)
# Adding the results to the dataframe
df = pd.concat([df, dummy], axis=1)
# Drop original variables
df = df.drop(['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
                                 'JobSatisfaction', 'WorkLifeBalance','BusinessTravel', 'Department',
                                 'Education','EducationField', 'Gender', 'JobLevel', 'JobRole',
                                 'MaritalStatus'], axis=1)

# 3. Recode attrition 
df['Attrition'] = df['Attrition'].replace({'Yes': 1, "No": 0})

# 4. Construct variables
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# 5. Check multicollinearity (because we will use logistic regression)
# Create VIF dataframe
from statsmodels.tools.tools import add_constant
X1 = add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns
  
# Calculating VIF for each feature
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(X1.columns)):
    vif_data.loc[vif_data.index[i],"VIF"] = variance_inflation_factor(X1.values, i)


print(vif_data)

# Drop those with VIF score over 5
X = X.drop(['Department_Research & Development','Department_Sales','EducationField_Life Sciences','EducationField_Marketing','EducationField_Medical','EducationField_Other','EducationField_Technical Degree'], axis=1)

# 5. Train-test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

# 6. Standardization using min max
from sklearn.preprocessing import MinMaxScaler
# Because there are many dummies (ranging from 0 to 1) we want to standardize the range for other variables to also range from 0 to 1  
X_train_std = X_train.copy()
features = X_train_std.iloc[:,0:11]
scaler = MinMaxScaler()
features = scaler.fit_transform(features.values)
X_train_std.iloc[:,0:11] = features

X_test_std = X_test.copy()
features = X_test_std.iloc[:,0:11]
features = scaler.transform(features.values)
X_test_std.iloc[:,0:11] = features



### Model 1: Logistic regression
# Train model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(X_train_std,y_train) 

# Print coefficients
pd.DataFrame(list(zip(X.columns,np.transpose(model.coef_))), columns = ['predictor','coefficient'])

# Evaluate model
y_test_pred = model.predict(X_test_std)

from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)
metrics.precision_score(y_test, y_test_pred)
metrics.recall_score(y_test, y_test_pred)
metrics.f1_score(y_test, y_test_pred)



### Model 2: Classification tree
# Hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth':[3,4,5,6],
              'min_samples_split': [3,4,5,6],
              'min_samples_leaf':[3,4, 5,6]}
ct = DecisionTreeClassifier()  

from sklearn.model_selection import GridSearchCV
model = GridSearchCV(ct, parameters, cv=5)

model.fit(X_train_std,y_train)

model.best_score_ 
model.best_params_ 

# Final model based on the tuned hyperparameters
ct = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=6)
model = ct.fit(X_train_std, y_train)

# Tree diagram
from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(150,50))
features = X.columns
classes = ['No','Yes']
tree.plot_tree(model, fontsize= 12, feature_names=features,class_names=classes,filled=True)
plt.savefig('out.pdf')
plt.show()

# Feature importance
pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])

# Make prediction and evaluate accuracy
y_test_pred = model.predict(X_test_std)

from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)
metrics.precision_score(y_test, y_test_pred)
metrics.recall_score(y_test, y_test_pred)
metrics.f1_score(y_test, y_test_pred)      



