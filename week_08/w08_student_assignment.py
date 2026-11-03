#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:53:10 2026

@author: lau
"""

#%% TERMINAL COMMAND

## conda activate methods3_2026
## pip install ucimlrepo

#%% LOAD WINE QUALITY

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 



#%% Exercise in class - overall aim: get the best classification,
# in terms of accuracy, on a test set using logistic regression
# There is a prize for the best classification found
# work in pairs or threes
# OPTIONAL: can you make an even better score with SVM?





## suggested operations in no particular order

#   a. test penalisers L1, L2 or elastic net
#   b. test different number of features in PCA space
#   c. scale variables
#   d. try different solvers (check the solver argument of LogisticRegression)
#   e. try multinomial or OvR classification
#   f. make sure to use the Pipeline function
#   g. make appropriate cross-validation (e.g. use GridSearchCV)
#   h. also show your confusion matrix
#   from sklearn.metrics import confusion_matrix
#   confmat = confusion_matrix(y_test, y_hat)
#   you can find the slides in github