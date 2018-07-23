# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 19:58:58 2018

@author: rpasr
"""

# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Procces data

dataSet = pd.read_csv("xp.csv")

x = dataSet.iloc[:,0].values
y = dataSet.iloc[:,-1].values

# split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
#TODO: Create Linear Regression

linReg = LinearRegression()

# fit the data 
X_train = np.reshape(X_train,[5,1])
y_train = np.reshape(y_train,[5,1])

linReg.fit(X_train,y_train)

# visuluize the information 

plt.scatter(X_test, y_test, color = 'red')
x = np.reshape(x,[8,1])
X_test = np.reshape(X_test,[3,1])
plt.plot(X_train,linReg.predict(X_train), color = "blue")










