from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas
import random
random.seed(2)

df = pandas.read_excel('data.xlsx')
data = df.as_matrix();

X = data[:, 0:1]
y = data[:, 1:2]

# X = np.random.rand(1000, 1)
# y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

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


# def grad(w):
# 	N = Xbar.shape[0]
# 	return 1/N * np.dot(Xbar.T, (Xbar.dot(w) - y))

# def cost(w):
# 	N = Xbar.shape[0]
# 	return .5/N * np.linalg.norm(y - Xbar.dot(w), 2) ** 2

# def numberical_grad(w, cost):
# 	eps = 1e-4
# 	g = np.zeros_like(w)
# 	for i in range(len(w)):
# 		w_p = w.copy()
# 		w_n = w.copy()
# 		w_p[i] += eps
# 		w_n[i] -= eps
# 		g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)
# 	return g

# def check_grad(w, cost, grad):
# 	w = np.random.rand(w.shape[0], w.shape[1])
# 	grad1 = grad(w)
# 	grad2 = numberical_grad(w, cost)
# 	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

# print('Check gradient ... ', check_grad(np.random.rand(2, 1), cost, grad))

# def myGD(w_init, eta):
# 	w = [w_init]
# 	for it in range(100):
# 		w_new = w[-1] - eta * grad(w[-1])
# 		if np.linalg.norm(grad(w_new)) / len(w_new) < 0.005:
# 			print("Break")
# 			break;
		
# 		w.append(w_new)
# 	return (w, it)

# w_init = np.array([[0.5], [-33]])
# (w1, it ) = myGD(w_init, 0.01)
# print(w1[-1])
# print('Solution found by GD : w = ', w1[-1].T, '\nAfter %d iterations.' %(it+1))
iterations = 1500
alpha = 0.01

## Add a columns of 1s as intercept to X. This becomes the 2nd column
X_df['intercept'] = 1

## Transform to Numpy arrays for easier matrix math
## and start beta at 0, 0
X = np.array(X_df)
y = np.array(y_df).flatten()
beta = np.array([0, 0])

def cost_function(X, y, beta):
    """
    cost_function(X, y, beta) computes the cost of using beta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y)

    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(beta)-y)**2)/2/m

    return J