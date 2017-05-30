from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

import random
random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

#building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w_rl = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula : w = ', w_rl.T)

#Display result
w = w_rl
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1 * x0

#Drawing the fitting line
plt.plot(X.T, y.T, 'b.') # data
plt.plot(x0, y0, 'y', linewidth = 2) # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()


def grad(w):
	N = Xbar.shape[0]
	return 1/N * np.dot(Xbar.T, (Xbar.dot(w) - y))

def cost(w):
	N = Xbar.shape[0]
	return .5/N * np.linalg.norm(y - Xbar.dot(w), 2) ** 2

def numberical_grad(w, cost):
	eps = 1e-4
	g = np.zeros_like(w)
	for i in range(len(w)):
		w_p = w.copy()
		w_n = w.copy()
		w_p[i] += eps
		w_n[i] -= eps
		g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)
	return g

def check_grad(w, cost, grad):
	w = np.random.rand(w.shape[0], w.shape[1])
	grad1 = grad(w)
	grad2 = numberical_grad(w, cost)
	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print('Check gradient ... ', check_grad(np.random.rand(2, 1), cost, grad))

def myGD(w_init, eta):
	w = [w_init]
	for it in range(100):
		w_new = w[-1] - eta * grad(w[-1])
		if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
			break;
		
		w.append(w_new)
	return (w, it)

w_init = np.array([[2], [1]])
(w1, it ) = myGD(w_init, 1)
print('Solution found by GD : w = ', w1[-1].T, '\nAfter %d iterations.' %(it+1))