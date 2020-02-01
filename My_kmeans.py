# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:39:43 2019

@author: Abhishek Goel
"""
# Implementation of K-Means Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the Dataset
dataset = pd.read_csv('Mall_Customers.csv')
print(dataset)
X = dataset.iloc[:,[3,4]].values

# import the kmeans library from sklearn
from sklearn.cluster import KMeans

# we need to find the optimum number of clusters first
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10,max_iter=300, random_state=0)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
  
print(wcss)

# plottting the Elbow Curve to select the optimum size of clusters
plt.plot(range(1,11), wcss,'blue')
plt.title('The Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# optimum number of clusters comes out to be 5 :))

# fitting actual model to 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# Lets do the Visualisation , beautiful thing now

# first see the Datapoints corresponding to different clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, color='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, color='cyan', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, color='magenta', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, color='green', label='Cluster 5')

# Now plot the centroids of the Clusters
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s=250, color='yellow', label='Centroid')
plt.title('K-Means Clustering: Unsupervised Learning')
plt.xlabel('Salaries in Rs.')
plt.ylabel('Shopping Score (1-100)')
plt.show()