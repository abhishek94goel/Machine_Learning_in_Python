# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 04:54:24 2019

@author: Abhishek Goel
"""
## Implementation of Polynomial Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Creating the Matrices
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# We dont need to split the train_test data here
# because we need all data from variable to fit our polyomial regression curve

# No feature scaling, since it is Polynomial regression

# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X,y)

### Fitting Polynomial Regression to our Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Now apply multiple linear regression to above polynomial fitted data
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualising the results of Simple Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg1.predict(X), c='blue')
plt.title('Truth or Bluff (Simple Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the results from Polynomial Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(X_poly), color = 'blue')
  # Generalization of above line is:
  # plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial regression model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

  # to improve the curve flow- use smaller steps in X coordinate
X_grid = np.arange(min(X), max(X), 0.1)  # output is matrix type, so change to array
X_grid = X_grid.reshape((len(X_grid),1))
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')

# Predicting a new result with Simple Linear Regression model
lin_reg1.predict(np.array([[6.5]]))

# Predicting a new result with Polynomial Regression model
lin_reg2.predict(poly_reg.fit_transform(np.array([[6.5]])))