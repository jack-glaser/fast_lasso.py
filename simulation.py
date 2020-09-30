import numpy as np
from fast_lasso import Lasso
import random as r
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

numobs = 100000
r.seed(69)
np.random.seed(69)

def makeVar(size): 
	var = np.zeros(numobs)
	for i in range(numobs):
		var[i] = r.uniform(-size, size)
	center = np.mean(var)
	var = var - center
	return var

x1 = makeVar(5)
x2 = makeVar(5)
x3 = makeVar(20)
x4 = 0.2*x1 -0.7*x3 + makeVar(2)
x5 = 0.6*x4 + makeVar(1)
x6 = -1.5*x2 + 0.8*x5 + makeVar(15)
x = np.transpose(np.vstack((x1,x2,x3,x4,x5,x6)))
# x = np.transpose(np.vstack((x1,x2,x3,x4)))
e = np.random.normal(0,1,numobs)

y = 3*x1 -17*x2 + 5*x3 + 6*x4 - 10*x5 + 2*x6 + e

# y = 3*x1 -17*x2 + 5*x3 + 6*x4 + e # 


lasso = Lasso()

# Example lambda path for our lasso
lamList = range(0, 500, 5)
lasso.lamdbaPath(x, y, lamList)

#Confirming the graph using the sklearn module, which implements
# the same algorithm as Hastie, Tibshirani and Friedman (see
# documentation here:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

lamList = np.flip(np.sort(lamList)) #go in decreasing order of lambdas
betaMatrix = np.zeros((np.size(x,1), len(lamList)))
for i, lda in enumerate(lamList):
	regL = linear_model.Lasso(alpha=lda, max_iter=100000)
	regL.fit(x, y)
	betaMatrix[:,i] = regL.coef_

plt.axhline(color='black')
plt.ylim([-20,10])
for row in betaMatrix:
	plt.plot(lamList, row)
plt.show()
