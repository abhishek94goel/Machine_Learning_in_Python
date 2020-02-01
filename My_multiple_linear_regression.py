# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:59:47 2019

@author: Abhishek Goel
"""
# Implementing the Multiple Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('50_Startups.csv')

# Seperate the matrices
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# turning categorial data to quantitave variable
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelencoder.fit(X[:,3])
X[:,3] = labelencoder.transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:,1:]

# Train test splitting of the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Fitting of Multiple variable regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the output on the test set
y_pred = regressor.predict(X_test)

### Building the optimal model using the Backward Elimination feature selection process
import statsmodels.formula.api as sm # this library does not work anymore
from statsmodels.regression import linear_model as lm

# we need to add one column of Ones in the begining of the X matrix
""" # This will append the column of ones at the end of X
X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)"""
# To insert the new ones column at the begining of X:
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# Now executing the backward elimination process
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = lm.OLS(y,X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = lm.OLS(y,X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()