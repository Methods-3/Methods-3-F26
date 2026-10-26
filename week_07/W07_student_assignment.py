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
            linear_model = None ## WHAT IS SUPPOSED TO BE HERE?

            # Apply sigmoid function to get probabilities
            yhat = None ## WHAT IS SUPPOSED TO BE HERE?
            
            self.cost_.append(cost_function(y, yhat))
   
            # Update weights using the gradient
            self.w_ += None ## WHAT IS SUPPOSED TO BE HERE?
            if self.n_iter_ > 1: ## we need at least two observations
                diff =None ## WHAT IS SUPPOSED TO BE HERE?
                # print(diff)
                if diff < self.tol:
                    break

    def predict(self, X):
        pass ## put in your own quantizer function if you feel up for it





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
