# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:10:56 2023

@author: ehan1
"""

# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
%matplotlib inline     
sns.set(color_codes=True)

# Importing dataset
car_df=pd.read_csv("automobile.csv")

# Checking the first five rows
car_df.head(5)

# Display all columns in the console
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (car_df.head(5))

# Checking data types
car_df.dtypes
car_df.info()

# Frequency tables for each categorical feature
pd.crosstab(index = car_df['Vehicle Size'], columns = 'counts')

for column in car_df.select_dtypes(include=['object']).columns:
    display(pd.crosstab(index=car_df[column], columns='% observations', normalize='columns')*100)

# Descriptive statistics for numerical features    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (car_df.describe())

# Some pre-processing
# 1. Duplicates?
# Checking duplicates
car_df[car_df.duplicated()].shape

# Drop duplicates
car_df = car_df.drop_duplicates()

car_df.shape

# 2. Irrelevant columns?
car_df = car_df.drop(columns=['Make', 'Model','Popularity'], axis=1)

# 3. Missing values?
# Checking missing values
print(car_df.isnull().sum())
car_df['Market Category'].unique()
car_df = car_df.drop(['Market Category'], axis=1)

# Drop missing values
car_df = car_df.dropna()

car_df.shape
print(car_df.isnull().sum())

# 4. Outliers?
# Check outliers using boxplot
sns.boxplot(x=car_df['Price'])
sns.boxplot(x="Price", y="Vehicle Size", data=car_df)

# Remove based on what criteria? 3 std above? interquartile range?
car_df['Price'].describe()
car_df['Price'].mean()+3*(car_df['Price'].std())

from scipy.stats import iqr
IQR = iqr(car_df['Price'])
q3 = car_df['Price'].quantile(.75)
new_df = car_df[~(car_df['Price'] > (q3 + 1.5 * IQR))]

sns.boxplot(x=new_df['Price'])

# 5. Dummify the categorical variables
# Using get_dummies
for column in car_df.select_dtypes(include=['object']).columns:
    dum = pd.get_dummies(car_df[column], prefix=column, drop_first=True)
    car_df = pd.concat([car_df, dum], axis=1)

# Using one hot encoder
from sklearn.preprocessing import OneHotEncoder

for column in car_df.select_dtypes(include=['object']).columns:
    car_df[column] = car_df[column].astype('category')
    car_df[column+'_new'] = car_df[column].cat.codes
    
dummy = OneHotEncoder(drop='first')
dummy_df = pd.DataFrame(dummy.fit_transform(car_df.iloc[:,12:]).toarray())
dummy_df.columns = dummy.get_feature_names_out()
new_df = car_df.join(dummy_df)

# 6. Distribution?
# Check distribution using histogram
sns.histplot(x = new_df['Price'])
sns.histplot(x = new_df['Price'], binwidth=10000)

# Further exploration
## Scatterplots
sns.regplot(x="highway MPG", y="Price", data=new_df)
plt.ylim(0,)

sns.regplot(x="Engine HP", y="Price", data=new_df)
plt.ylim(0,)

## Correlation
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (new_df.corr())

c= new_df.corr()
sns.heatmap(c,cmap="rocket_r",annot=True)
