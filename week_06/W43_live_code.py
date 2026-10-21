#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:54:12 2025

@author: lau
"""

#%% IMPORTS

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid', context='notebook')   
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap

#%%# Sklearn's implementation


# from sklearn.linear_model import Perceptron

# print(load_iris(as_frame=True).keys())
# target_names = load_iris(as_frame=True)['target_names']
# feature_names = load_iris(as_frame=True)['feature_names']

# X_subset = X[y < 2, 0:3:2]  ## petal length and sepal length
# y_subset = y[y < 2]

# clf = Perceptron()
# clf.fit(X_subset, y_subset)
# print(clf.score(X_subset, y_subset))

#%% PERCEPTRON EXAMPLE
#%%# Raschka's implementation

X, y = load_iris(return_X_y=True)
X_subset = X[y < 2, 0:3:2]  ## petal length and sepal length
y_subset = y[y < 2]
y_subset[y_subset == 0] = -1

import numpy as np

## old implementation (2021)
class Perceptron(object):
    """ Perceptron classifier
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
        
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
        
    """
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Traing vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            self.plot_decision_regions_adapted(X, y)
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                
                self.w_[1:] += update * xi
                self.w_[0] += update
                # print(update)
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print(errors)
        self.score(X, y)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
    ## added by lau
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.zeros(shape=len(predictions))
        for prediction_index, prediction in enumerate(predictions):
            if prediction == -1 and y[prediction_index] == -1:
                accuracy[prediction_index] = 1
            if prediction == 1 and y[prediction_index] == 1:
                accuracy[prediction_index] = 1
        return accuracy
    
    def plot_decision_regions_adapted(self, X, y,  resolution=0.02):
        # setup marker generator and color map
        plt.figure()
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        x1_min, x1_max  = (-5, 10)
        x2_min, x2_max  = (-5, 10)
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.xlabel('Petal length (cm)')
        plt.ylabel('Sepal length (cm)')
        # plt.ylim(-10, 10)
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=cmap(idx),
            marker=markers[idx], label=cl)
        plt.show()
        

ppn = Perceptron(eta=0.01)
ppn.fit(X_subset, y_subset)
pred = ppn.predict(X_subset)
scores = ppn.score(X_subset, y_subset)
overall_score = np.sum(scores) / len(scores)
print(overall_score)




#%%# Raschka's ADALINE implementation

X, y = load_iris(return_X_y=True)

X_subset = X[y < 2, 0:3:2]  ## petal length and sepal length
y_subset = y[y < 2]
y_subset[y_subset == 0] = -1

class AdalineGD(object):
    """ ADAptive LInear NEuron classifier
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
        
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
        
    """
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y, plot_every=None, stop_on_iter=False):
        """ Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Traing vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for self.i in range(self.n_iter):
            if plot_every is not None:
                if self.i % plot_every == 0 or self.i == (self.n_iter - 1):
                    self.plot_decision_regions_adapted(X, y,
                                                       plot_every=plot_every)
                   
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            if stop_on_iter:
                print(f'After iteration {self.i}\n'
                      f'Weights: constant weight (w0) (bias): {self.w_[0]}\n'
                      f'Linear weights (w): {self.w_[1:]}')
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            print(f'J(w) = {cost}\n')
        return self
            
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Computer linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
    ## added by lau
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.zeros(shape=len(predictions))
        for prediction_index, prediction in enumerate(predictions):
            if prediction == -1 and y[prediction_index] == -1:
                accuracy[prediction_index] = 1
            if prediction == 1 and y[prediction_index] == 1:
                accuracy[prediction_index] = 1
        return accuracy
    
    def plot_decision_regions_adapted(self, X, y, plot_every, resolution=0.02):
        # setup marker generator and color map
        plt.figure()
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        x1_min, x1_max  = (-5, 10)
        x2_min, x2_max  = (-5, 10)
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.xlabel('Petal length (cm)')
        plt.ylabel('Sepal length (cm)')
        
        plt.title(f'Decision line after {self.i} iterations')
        
        # plt.ylim(-10, 10)
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=cmap(idx),
            marker=markers[idx], label=cl)
        plt.show()


# ada = AdalineGD()
ada = AdalineGD(eta=0.0001, n_iter=1000) ## 10 100 1000
# ada.fit(X_subset, y_subset, plot_every=1)
ada.fit(X_subset, y_subset, plot_every=100, stop_on_iter=True) # 1 10 100
pred = ada.predict(X_subset)
scores = ada.score(X_subset, y_subset)
overall_score = np.sum(scores) / len(scores)
print(overall_score)
plt.figure()
plt.plot(ada.cost_)
plt.show()
