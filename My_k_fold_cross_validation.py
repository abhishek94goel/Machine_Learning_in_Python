# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 04:42:35 2019

@author: Abhishek Goel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

# features scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting the kernel SVM classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=44)
classifier.fit(X_train, y_train)

# Predicting the results
y_pred = classifier.predict(X_test)

# Accuracy score and confusion metrics on test set
from sklearn.metrics import accuracy_score, confusion_matrix
acc_s = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Applying the K-Fold Cross Validation Technique
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y=y_train, cv=20, n_jobs=-1)
accuracies.mean()
accuracies.std()

# average accuracy of 90 % in 20 folds.
# Standard deviation of accuracy results = 7% (good results)