#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:48:37 2024

@author: lau
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, yhat):
    return -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

def gradient(X, y, yhat):
    return np.dot((y - yhat), X) 


class LogisticRegressionGD:
    def __init__(self, eta=0.01, n_iterations=1000, tol=1e-4):
        self.eta = eta
        self.n_iterations = n_iterations
        self.w_ = None
        self.cost_ = []
        self.tol = tol
        self.n_iter_ = 0

    def fit(self, X, y):

        X = np.insert(X, 0, 1, axis=1) # adding intercept term
        self.w_ = np.zeros(X.shape[1])
        # Gradient Descent
        for _ in range(self.n_iterations):
            self.n_iter_ += 1
            # Calculate linear combination of features and weights
            # linear_model = np.dot(X[:, 1:], self.w_[1:]) + self.w_[0]
            linear_model = np.dot(X, self.w_)

            # Apply sigmoid function to get probabilities
            yhat = sigmoid(linear_model)
            
            self.cost_.append(cost_function(y, yhat))
   
            # Update weights using the gradient
            self.w_ += self.eta * gradient(X, y, yhat)
            # self.w_[0] += self.eta * (y-yhat).mean()
            if self.n_iter_ > 1: ## we need at least two observations
                diff = abs(self.cost_[-2] - self.cost_[-1])
                # print(diff)
                if diff < self.tol:
                    break

    def predict(self, X):
        # Add intercept term to the input features
        # Compute linear model
        linear_model = np.dot(X, self.w_[1:]) + self.w_[0]
        # Apply sigmoid to get predicted probabilities
        self.probabilities = sigmoid(linear_model)
        # Convert probabilities to binary predictions
        return [1 if i > 0.5 else 0 for i in self.probabilities]


X, y = load_iris(return_X_y=True)
X = X[y < 2, 0:3:2]  ## petal length and sepal length
y = y[y < 2]

logreg = LogisticRegressionGD(eta=0.001, n_iterations=10000, tol=1e-4)
logreg.fit(X, y)

plt.figure()
plt.xlabel('Iteration (#)')
plt.ylabel('Cost (J(w))')
plt.title('Converged at iteration: ' + str(logreg.n_iter_))
plt.plot(logreg.cost_)
plt.show()

print(logreg.predict(X))
print(logreg.predict(X) == y)
