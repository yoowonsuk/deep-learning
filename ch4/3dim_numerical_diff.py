import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val -h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def function_2(x):
    return np.sum(x**2) # return x[0]**2 + x[1]**2

a = numerical_gradient(function_2, np.array([3., 4.]))
print(a)
a = numerical_gradient(function_2, np.array([0., 2.]))
print(a)
a = numerical_gradient(function_2, np.array([3., 0.]))
print(a)

