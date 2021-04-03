# -*- coding: utf-8 -*-

import operator as op
import numpy as np
import matplotlib.pyplot as plt
import math

m = 100
X = np.random.rand(m, 1) * 2
y = np.sin(2 * math.pi * X) + np.random.randn(m, 1)
epochs = 100
L = 0.4
tau = 0.1

def sort(X, y):  # sorting the data generated, namely, X and y 
    sorted_zip = sorted(zip(X, y), key = op.itemgetter(0))
    X, y = zip(*sorted_zip)
    return X, y

def weightedLinearRegression(X, y, iterNo, eta, x, tau):
    weights = np.random.random(m)  # keeping the weights of each x
    theta0 = np.random.random(1)
    theta1 = np.random.random(1)
    sum0 = 0
    sum1 = 0
    
    # weight calculation by the formula
    for w in range(m):
        weights[w] = math.exp(-1 * (((X[w] - x) ** 2) / (2 * (tau ** 2))))

    for j in range(iterNo):
        for i in range(m):
            hypothesis = np.dot(X[i], theta1)
            hypothesis += theta0
            loss = hypothesis - y[i]
            gradient0 = (loss * weights[i])
            gradient1 = (loss * X[i] * weights[i])
            sum0 += gradient0
            sum1 += gradient1

        sum0 *= 2.0 / m
        sum1 *= 2.0 / m
        theta0 -= eta * sum0
        theta1 -= eta * sum1
    return theta0, theta1

y_predicted = np.random.rand(m, 1)  # holding yhe y values which are predicted

# calling the weighted linear regression
for i in range(m):
    theta0, theta1 = weightedLinearRegression(X, y, epochs, L, X[i], tau)
    y_predicted[i] =  theta1 * X[i]
    y_predicted[i] += theta0
    
X, y_predicted = sort(X, y_predicted) # # In order to plot the multiple data, it is essential to sorting 

# plotting
plt.title('Weighted Linear Regression w/ tau = 0.1')
plt.scatter(X, y)
plt.plot(X, y_predicted)
plt.show()
plt.savefig(r"C:\Users\pc\Desktop\tau0.1.png")
