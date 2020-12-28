import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('mlm.csv', delimiter=',', skiprows=1)
X = np.insert(dataset[:, :2], 0, values=1, axis=1)
y = dataset[:, 2:].squeeze()


class Solution:
	def __init__(self, n):
		self.n = n
		self.W = np.random.rand(n)

	def onestep(self, X, y):
		m = X.shape[0]
		y_pred = np.dot(X, self.W)
		loss = 0.5*np.sum((y_pred - y)**2) / m
		dW = np.dot(X.T, y_pred - y) / m
		return loss, dW

	def train(self, X, y, stepSize, trainTimes):
		for i in range(trainTimes):
			loss, dW = self.onestep(X, y)
			self.W -= stepSize * dW
			if (i+1) % 5000 == 0:
				print('step = '+str(i+1)+' loss = '+str(loss))


sol = Solution(X.shape[1])
sol.train(X, y, 1e-4, int(1e6))
print('W: '+str(sol.W))

ax = plt.gca(projection='3d')
point_x = dataset[:, 0]
point_y = dataset[:, 1]
point_z = dataset[:, 2]
xa = np.linspace(0, 100, 5)
ya = np.linspace(0, 100, 5)
sur_x, sur_y = np.meshgrid(xa, ya)
ax.plot_surface(
	sur_x, sur_y, sol.W[0]+sur_x*sol.W[1]+sur_y*sol.W[2], color='yellow', alpha=0.6)
ax.scatter3D(point_x, point_y, point_z, cmap='Blues')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
