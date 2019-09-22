import numpy as np

print(np.float32(1e-50))

print(np.float32(1e-4))

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h)-f(x-h)) / (2 * h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

import matplotlib.pylab as plt

x = np.arange(.0, 20., .1) # from 0 to 20 at intervals of 0.1
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
# plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) -d*x
    return lambda t: d*t + y

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y2)
# plt.show()

def fuction_2(x):
    return x[0]**2 + x[1]**2 # return np.sum(x**2), and consider x numpy array

