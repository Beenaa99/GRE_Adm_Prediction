# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:43:41 2020

@author: nomka
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Admission_Predict.csv')

del dataset['Serial No.'] 
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_Y = StandardScaler()

#Y_train = sc_Y.fit_transform(Y_train.reshape(-1,1))
#Y_train = sc_Y.fit_transform(Y_train)
#Y_test = sc_Y.fit_transform(Y_test.reshape(-1,1))

from sklearn.linear_model import LinearRegression
regg = LinearRegression()
regg.fit(X_train,Y_train)

Y_pred = regg.predict(X_test)

import statsmodels.api as sm
X = np.append(arr = np.ones((400,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5,6]]
regg_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regg_OLS.summary()