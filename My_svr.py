# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:44:43 2019

@author: Abhishek Goel
"""
# Implementation of SVR

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# creating the matrices
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# features scaling is required, SVR do not included default features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#### Preprocessing is complete here, Now we evaluate our SVR #####
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# Lets visualize the SVR results
plt.scatter(X, y, c = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# evaluate the prediction on a particular value
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
y_pred