# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 05:39:51 2019

@author: Abhishek Goel
"""
# Hierarchical Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Evaluating the Dendrograms using scipy library
import scipy.cluster.hierarchy as sch
dedrograms = sch.dendrogram(sch.linkage(X,method='ward', metric='euclidean'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# On assessment, we get 5 clusters to be formed

# now, actually implementing the Hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_pred_hc = hc.fit_predict(X)
#print(y_pred_hc)

# Visualizing the results
plt.scatter(X[y_pred_hc==0, 0], X[y_pred_hc==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_pred_hc==1, 0], X[y_pred_hc==1, 1], s=100, c='cyan', label='Cluster 2')
plt.scatter(X[y_pred_hc==2, 0], X[y_pred_hc==2, 1], s=100, c='magenta', label='Cluster 3')
plt.scatter(X[y_pred_hc==3, 0], X[y_pred_hc==3, 1], s=100, c='green', label='Cluster 4')
plt.scatter(X[y_pred_hc==4, 0], X[y_pred_hc==4, 1], s=100, c='blue', label='Cluster 5')
plt.title('Hierarchical Clustering')
plt.xlabel('Income in Rs')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()