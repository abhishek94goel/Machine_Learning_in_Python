# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:44:00 2019

@author: Abhishek Goel
"""
# Random Forest Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Creating the matrices
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# implementing the Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 44, random_state = 0)
regressor.fit(X,y)

# Predicting a particular input value
display(regressor.predict(np.array([[6.5]])))

# Visualizing the output
plt.scatter(X,y, c='red')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression- Prediction of Salary based on Positions')
plt.xlabel('Position Level from 1 to 10')
plt.ylabel('Salary')
plt.show()
plt.show()