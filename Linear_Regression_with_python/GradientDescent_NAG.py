# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2 * np.random.rand(1000, 1)

#Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)

def cost(w):
	return .5 / Xbar.shape[0] * np.linalg.norm(y - Xbar.dot(w), 2) ** 2

def grad(w):
	return 1/Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)

def numerical_grad(w, cost):
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
	grad2 = grad(w)
	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print('Checking gradient ... ', check_grad(np.random.rand(2, 1), cost, grad))

def GD_NAG(w_init, grad, eta, gamma):
	w = [w_init]
	v = [np.zeros_like(w_init)]
	for it in range(100):
		v_new = gamma * v[-1] + eta * grad(w[-1] - gamma * v[-1])
		w_new = w[-1] - v_new
		if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
			break

		w.append(w_new)
		v.append(v_new)
	return(w, it)

w_init = np.array([[2], [1]])
(w_mm, it_mm) = GD_NAG(w_init, grad, .5, .9)
print(w_mm[-1], '\n\n', it_mm)

