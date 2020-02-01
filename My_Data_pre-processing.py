# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:24:13 2019

@author: Abhishek Goel
"""

###### Data Pre-processing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the Dataset
dataset = pd.read_csv('Data.csv')

# creating the matrix
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# taking care of the missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorial data

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])   # to make dummy encoding of categorial data
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into training set and testing set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)