import pickle
import numpy as np
import matplotlib.pyplot as plt

class_dir = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
dataset = np.loadtxt('iris.csv', delimiter=',', skiprows=1,
					 converters={4: lambda s: class_dir[s]})
X, Y = dataset[:, :4], dataset[:, 4:].squeeze().astype(int)


class Solution:
	def __init__(self, n, c):  # n维数 c分类数
		self.n = n
		self.c = c
		self.W1 = np.random.rand(n, c) - 0.5
		self.W2 = np.random.rand(c, c)
		self.B1 = np.random.rand(c)
		self.B2 = np.random.rand(c)

	def __Softmax(self, Z, Y):
		S = np.sum(np.exp(Z), axis=1).repeat(Z.shape[1]).reshape(Z.shape)
		res = np.exp(Z) / S
		return res

	def onestep(self, X, Y):
		m = X.shape[0]
		Z1 = np.dot(X, self.W1) + self.B1
		A1 = np.maximum(Z1, 0)
		Z2 = np.dot(A1, self.W2) + self.B2
		A2 = self.__Softmax(Z2, Y)
		loss = np.sum(-np.log(A2[np.arange(m), Y]))
		dZ2 = A2
		dZ2[np.arange(m), Y] -= 1
		dW2 = np.dot(A1.T, dZ2)
		dB2 = np.mean(dZ2, axis=0)
		dZ1 = np.dot(dZ2, self.W2.T) * (Z1 > 0)
		dW1 = np.dot(X.T, dZ1)
		dB1 = np.mean(dZ1, axis=0)
		return loss, dW1, dW2, dB1, dB2

	def train(self, X, Y, stepsize, steps):
		for i in range(steps):
			loss, dW1, dW2, dB1, dB2 = self.onestep(X, Y)
			if i == 0 or (i+1) % 5000 == 0:
				print('step '+str(i+1)+':\tloss='+str(loss))
			self.W1 -= stepsize * dW1
			self.W2 -= stepsize * dW2
			self.B1 -= stepsize * dB1
			self.B2 -= stepsize * dB2

	def test(self, X, Y):
		Z1 = np.dot(X, self.W1) + self.B1
		A1 = np.maximum(Z1, 0)
		Z2 = np.dot(A1, self.W2) + self.B2
		A2 = self.__Softmax(Z2, Y)
		accuracy = np.sum(np.argmax(A2, axis=1) == Y) / Y.shape[0]
		print('accuracy = ' + str(accuracy * 100) + '%')


savefile = open('iris.data', 'rb')
sol = pickle.load(savefile)
savefile.close()

'''
sol = Solution(X.shape[1], 3)
sol.train(X, Y, 1e-5, int(1e5))

savefile = open('iris.data', 'wb')
pickle.dump(sol, savefile)
savefile.close()
'''

sol.test(X, Y)
