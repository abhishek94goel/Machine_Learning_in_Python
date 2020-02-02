# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 06:12:11 2019

@author: Abhishek Goel
"""
# Lets implement the Grid Search algo for Kernel SVM model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.drop(['User ID','Gender','Purchased'],axis=1)
Y = dataset['Purchased']

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# applying the Kernel SVM classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=44)
classifier.fit(X_train, y_train)

# Predicting the results
y_pred = classifier.predict(X_test)

# applying K-fold Crosss validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y=y_train, cv=20, n_jobs=-1)
print(accuracies.mean())
print(accuracies.std())

# Applying the Grid search for model parameters evaluation
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 0.10, 10, 0.01, 0.5, 1.5, 2.0, 3.0, 4.0], 'kernel':['linear']},
              {'C':[1, 0.10, 10, 0.01, 0.5, 1.5, 2.0, 3.0, 4.0], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
              ]

grid_search = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy',
                        n_jobs=-1, cv=30)

grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_