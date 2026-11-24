#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:10:16 2024

@author: lau
"""

## EXERCISE:
    
## 1) Finish the kmeans class below, filling in 
## "initalise_centroids", 
## "assign_to_centroids",
## "move_centroids_to_new_centres",
## "calculate_cost"

## 2) Fit the kmeans for the X data and plot it

#%% kmeans class

import numpy as np
import sys

class kmeans(object):
        
    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.max_iter         = max_iter
        self.tol              = tol
        self.k                = k
        self.centroids_       = None # this is where the fitted centroids will be
        self.centroid_labels_ = None # this is where the fitted centrod labels will be
        self.cost_            = list() # empty list, where the cost for each iteration can go
       
    def initialise_centroids(self, X):
        ## 1) Randomly pick k centroids from the sample points
        ## as initial cluster centers
        ## have a look at np.random.choice to choose random starters
        
        initial_centroids = None # change this
        return initial_centroids
    
    def assign_to_centroids(self, X, centroids):
        ## 2) Assign each sample to the nearest centroid μ(j), j ∈ {1, ... , k}
        ## have a look at np.linalg.norm to calculate the distance from which
        ## you can get the centroid label
        
        n_samples = X.shape[0]
        centroid_labels = np.zeros(n_samples) ## fill in labels (0, 1 ,2)

        return centroid_labels
    
    def move_centroid_to_new_centres(self, X, centroid_labels):
        ## 3) Move the centroids to the center of the samples
        ## that were assigned to it
        
        new_centroids = np.zeros(shape=(self.k, X.shape[1])) # fill coordinates
        
        return new_centroids
    
    def calculate_cost(self, X, centroids, centroid_labels):
        ## squared Euclidean distance        
        SSE = None ## fill in this
        return SSE
    
    def fit(self, X):
        ## 4) Repeat the steps 2 and 3
        ## until the cluster assignment does not change
        ## or a user-defined tolerance
        ## or a maximum number of iterations is reached
        
        ## step 1)
        centroids = self.initialise_centroids(X)
        self.centroids_ = np.expand_dims(np.array(centroids), axis=0)
        print(centroids)
        
        for n_iter in range(self.max_iter):
            
            ## printing progress
            sys.stderr.write('\rIteration: %d/%d' % (n_iter+1, self.max_iter))
            sys.stderr.flush()
            
            ## step 2)
            centroid_labels = self.assign_to_centroids(X, centroids)
            if n_iter == 0: ## initial labels
                self.centroid_labels_ = \
                    np.expand_dims(np.array(centroid_labels), axis=0)
                self.cost_.append(self.calculate_cost(X, centroids,
                                                      centroid_labels))
            ## step 3)
            centroids = self.move_centroid_to_new_centres(X, centroid_labels)
            
            ## build kmeans object
            self.cost_.append(self.calculate_cost(X, centroids,
                                                  centroid_labels))
            self.centroids_ = np.concatenate((self.centroids_, 
                                      np.expand_dims(centroids, axis=0)),
                                                axis=0)
           
            self.centroid_labels_ = np.concatenate((self.centroid_labels_,
                      np.expand_dims(centroid_labels, axis=0)), axis=0)
            
            ## convergence check
            ## break out of loop if tol is reached
            if n_iter > 1:
                diff = np.abs(self.cost_[-2] - self.cost_[-1])
                if diff < self.tol:
                    break
        
#%% fit data (X)

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)               

km = kmeans(k=3)

km.fit(X)

y_km            = km.centroid_labels_[-1, :] ## pick last one (converged one)
cluster_centers = km.centroids_[-1, :, :] # pick last one (converged one)

#%% plot fitted clusters

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')
plt.scatter(cluster_centers[:, 0],
            cluster_centers[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

