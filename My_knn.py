# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:30:50 2019

@author: Abhishek Goel
"""
# KNN Classifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the Dataset
dataset = pd.read_csv('C:\\Users\\Abhishek Goel\\Documents\\GitHub_repos\\Machine_Learning_in_Python\\Social_network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# Train Test split the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

# Features Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting KNN to the training Set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the output on the test set
y_pred = classifier.predict(X_test)

# Confusion Matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy Score
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Accuracy is = ' + (acc*100).astype(str) + '%')

# Visualising the Training Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=min(X_set[:,0])-1, stop=max(X_set[:,0])+1, step=0.01) , 
                     np.arange(start=min(X_set[:,1])-1, stop=max(X_set[:,1])+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X2.shape), alpha = 0.75,
             cmap = ListedColormap(('red','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'yellow'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=min(X_set[:,0])-1, stop=max(X_set[:,0])+1, step=0.01) , 
                     np.arange(start=min(X_set[:,1])-1, stop=max(X_set[:,1])+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X2.shape), alpha = 0.75,
             cmap = ListedColormap(('red','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'yellow'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()