import numpy as np
import matplotlib.pyplot as plt
import time

class Lasso(object): 

	def __init__(self):
		pass

	def train(self, X, Y, lam):

		#Warm starts with previous betas if previously defined, else calls warmstart func
		try: 
			self.betas
		except AttributeError:
			self.warmStart(X, Y)
		
		#precomputes 

		(sqNorms, yx, x1, xx) = self.precompute(X,Y)

		converged = False
		iters = 0
		while not converged:
			iters += 1
			preLoss = self.loss(X,Y)

			# adjusting betas
			for i in range(self.numPredictors):
				sumTerm = 0
				for j in range(self.numPredictors):
					sumTerm += self.betas[j]*xx[i,j] #note that xx[i,j] == 0 if i === j
				zx = yx[i] - self.b0*x1[i] - sumTerm

				bHat = zx/sqNorms[i]
				gamma = (len(Y)*lam)/sqNorms[i]
				self.betas[i] = self.softThresh(bHat, gamma)

			# adjusting beta0
			meanPred = 0
			for i in range(self.numPredictors):
				meanPred += self.betas[i]*np.mean(X[:,i])
			self.b0 = np.mean(Y) - meanPred

			# checking for convergence
			postLoss = self.loss(X,Y)
			if abs(preLoss - postLoss) < 0.000001 or iters > 10000:
				converged = True


	def lamdbaPath(self, X, Y, lamList):
		lamList = np.flip(np.sort(lamList)) #go in decreasing order of lambdas
		betaMatrix = np.zeros((np.size(X,1), len(lamList)))
		for i, lda in enumerate(lamList):
			self.train(X,Y, lda)
			betaMatrix[:,i] = self.betas

		plt.axhline(color='black')
		plt.ylim([-20,10])
		for row in betaMatrix:
			plt.plot(lamList, row)
		plt.show()

	def precompute(self, X, Y):
		sqNorms = np.zeros(self.numPredictors)
		for i in range(self.numPredictors):
			sqNorms[i] = np.matmul(X[:,i], X[:,i])

		yx = np.zeros(self.numPredictors)
		for i in range(self.numPredictors):
			yx[i] = np.inner(Y, X[:, i])

		x1 = np.zeros(self.numPredictors)
		for i in range(self.numPredictors):
			x1[i] = np.sum(X[:,i])

		xx = np.zeros((self.numPredictors, self.numPredictors))
		for i in range(self.numPredictors):
			for j in range(self.numPredictors):
				if i == j:
					xx[i,j] = 0
				else:
					xx[i,j] = np.matmul(X[:,j], X[:,i])

		return (sqNorms, yx, x1, xx)

	def softThresh(self, bHat, gamma):
		return np.sign(bHat)*max((abs(bHat) - gamma), 0)

	def warmStart(self, X, Y): 
		self.numPredictors = np.size(X, 1)
		self.b0 = 0
		self.betas = np.zeros(self.numPredictors)

	def predict(self, X):
		return np.matmul(X, self.betas) + self.b0

	def loss(self, X, Y):
		predictions = self.predict(X)
		return np.mean(np.absolute(Y - predictions))