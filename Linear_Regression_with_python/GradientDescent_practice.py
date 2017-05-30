from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas

def grad(Xbar, y, w):
	N = Xbar.shape[0]
	return 1/N * np.dot(Xbar.T, (Xbar.dot(w) - y))

def cost(Xbar, y, w):
		N  = Xbar.shape[0]
		return .5/N * np.linalg.norm(y - Xbar.dot(w), 2) ** 2

def numberical_grad(Xbar, y, w, cost):
		eps = 1e-4
		g = np.zeros_like(w)
		for i in range(len(w)):
			w_p = w.copy()
			w_n = w.copy()
			w_p[i] += eps
			w_n[i] -= eps
			g[i] = (cost(Xbar, y, w_p) - cost(Xbar, y, w_n)) / (2 * eps)
		return g

def check_grad(w, cost, grad):
		w = n.random.rand(w.shape[0], w.shape[1])
		grad1 = grad(w)
		grad2 = numberical_grad(w, cost)
		return True if np.linalg.norm(grad1 - grad2) < 1e-6	else False

def myGD(Xbar, y, w_init, eta):
	w = [w_init]
	for it in range(100):
		w_new = w[-1] - eta * grad(Xbar, y, w[-1])
		if np.linalg.norm(grad(Xbar, y, w_new)) / len(w_new) < 1e-3:
			break
		w.append(w_new)
	return (w, it)

def has_converged(Xbar, y, theta_new):
	return np.linalg.norm(grad(Xbar, y, theta_new) / len(theta_new)) < 1e-3

def GD_momentum(Xbar, y, theta_init, eta, gamma):
	theta = [theta_init]
	v_old = [np.zeros_like(theta_init)]

	for it in range(100):
		v_new = gamma * v_old[-1] + eta * grad (Xbar, y, theta[-1])
		theta_new = theta[-1] - v_new
		if has_converged(Xbar, y, theta_new):
			break
		theta.append(theta_new)
		v_old.append(v_new)
	return (theta, it)

X = np.random.rand(1000, 1)
y = 5 + 4*X + .2 * np.random.randn(1000, 1)

ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones, X), axis = 1)

w_init = np.array([[2], [1]])
theta_init = np.array([[1], [2]])

(w, it) = GD_momentum(Xbar, y, w_init, 0.5, 0.5)
print(w[-1], it)





