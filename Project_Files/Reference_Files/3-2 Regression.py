# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:48:13 2023

@author: ehan1
"""
################# Before model building ######################################
# Import data
import pandas as pd
df = pd.read_csv("cereals.CSV")

# Construct variables (Calories, protein, fat, sodium, fiber as predictors; Rating as a target)
X = df.iloc[:,3:8] #iloc is specifically for slicing the dataframe
y = df['Rating']


## Detecting multicollinearity
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


## Standardize the predictors
X.describe()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns=X.columns) # transform into a dataframe and add column names


################# Model building ######################################
## 1. Linear regression from a statistics perspective (statsmodels)
# Load libraries
import statsmodels.api

# Add constant
X2 = add_constant(scaled_X)

# Run regression
regression = statsmodels.api.OLS(y,X2)
model = regression.fit()

# Show results
model.summary()


## 2. Linear regression from a data mining perspective (sklearn)
# Load libraries
from sklearn.linear_model import LinearRegression

# Run linear regression
lm1 = LinearRegression() # create a blank model object and instantiate the model
model1 = lm1.fit(scaled_X, y) # fit the regression model using the predictors X and the target y

# View results
model1.intercept_
model1.coef_


## 3. Cross-Validation 1 (Validation set approach)
# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.33, random_state = 5)

# Run linear regression based on the training data
lm2 = LinearRegression()
model2 = lm2.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = model2.predict(X_test)
# y_test_preddf=pd.DataFrame(y_test_pred, columns=['Predicted rating']) if need to convert to dataframe

# Calculate the MSE for the test set
from sklearn.metrics import mean_squared_error
lm2_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using validation set approach = "+str(lm2_mse))


## 4. Cross-Validation 2 (k-fold cv)
# Develop Linear Regression Model
lm3 = LinearRegression()

from sklearn.model_selection import cross_val_score
score = cross_val_score(lm3, scaled_X, y, scoring='neg_mean_squared_error',cv=5)
# scoring for defining a performance measure you are going to use (in our case, MSE)
# cv for setting the number of folds. in other words, it is k from k-fold cv.
print(score) # score will have five elements, corresponding to the MSE calculated from each iteration
print("Test MSE using k-fold CV with 5 folds = "+str(score.mean())) # to create a final, aggregate measure, you average the five MSEs

# Testing different values of K
for i in range (3,11):
    score = cross_val_score(lm3, scaled_X, y, scoring='neg_mean_squared_error',cv=i)
    print("Test MSE using k-fold CV with ",i," folds = "+str(score.mean()))


## 5. Ridge regression
from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 1
ridge = Ridge(alpha=1)
ridge_model = ridge.fit(X_train,y_train)

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(ridge_mse))

# What if penalty = 0?

# Finding optimal penalty
for i in range (1,10):
    ridge3 = Ridge(alpha=i)
    ridge_model3 = ridge3.fit(X_train,y_train)
    y_test_pred3 = ridge_model3.predict(X_test)
    print('Alpha = ',i,' / MSE =',mean_squared_error(y_test, y_test_pred3))


## 6. LASSO
from sklearn.linear_model import Lasso

# Run LASSO with penalty = 1
lasso = Lasso(alpha=1)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred4 = lasso_model.predict(X_test)

# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred4)
print("Test MSE using lasso with penalty of 1 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_

# Finding optimal alpha
for i in range (1,10):
    lasso = Lasso(alpha=i)
    lasso_model = lasso.fit(X_train,y_train)
    y_test_pred5 = lasso_model.predict(X_test)
    print('Alpha = ',i,' / MSE =',mean_squared_error(y_test, y_test_pred5))
    
    
## Why ridge and LASSO didn't improve?    
from mlxtend.evaluate import bias_variance_decomp
X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()
model = LinearRegression()
mse, bias, var = bias_variance_decomp(model, X_train_array, y_train.values, X_test_array, y_test.values, loss='mse', num_rounds=200, random_seed=1)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
    