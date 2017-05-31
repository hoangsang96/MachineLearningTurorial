from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)

def cost(w):
	return .5 / Xbar.shape[0] * np.linalg.norm(y - Xbar.dot(w), 2 ) ** 2

def grad(w):
	return 1 / Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)

def numerical_grad(w, cost):
	eps = 1e-4
	g = np.zeros_like(w)
	for i in range(len(w)):
		w_p = w.copy()
		w_n = w.copy()
		w_p[i] += eps
		w_n[i] -= eps
		g[i] = (cost(w_p) - cost(w_n)) / (2*eps)
	return g

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 

#single point gradient
def sgrad(w, i, rd_id):
	true_i = rd_id[i];
	xi = Xbar[true_i, :]
	yi = y[true_i]
	a = np.dot(xi, w) - yi
	return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
	w = [w_init]
	w_last_check = w_init
	iter_check_w = 10
	N = X.shape[0]
	count = 0
	for it in range(10):
		#shuffle data
		rd_id = np.random.permutation(N)
		for i in range(N):
			count += 1
			g = sgrad(w[-1], i, rd_id)
			w_new = w[-1] - eta * g
			w.append(w_new)
			if count % iter_check_w == 0:
				w_this_check = w_new
				if np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-3:
					return (w)

				w_last_check = w_this_check
	return (w)

X = np.random.rand(1000, 1)
y = 4 + 3*X + .2 * np.random.randn(1000, 1)

ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones, X), axis = 1)

w_init = np.array([[2], [1]])

w = SGD(w_init, grad, .1)
print(len(w), '\n\n', w)