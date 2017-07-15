from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
import pandas

df = pandas.read_excel('data.xlsx')
data = df.as_matrix()

X = data[:, 0:1]
y = data[:, 1:2]

plt.plot(X , y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
#plt.show()

ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

print(w)

w_0 = w[0][0]
w_1 = w[1][0]

plt.hold(True)
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0
plt.plot(x0, y0, hold=True)

# plt.show()
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0
print(w_1, " ", w_0)
print( 'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( 'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )