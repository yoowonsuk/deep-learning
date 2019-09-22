import numpy as np
import gradient_2d

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = gradient_2d.numerical_gradient(f, x)
        x -= lr * grad
        
    return x

def function_2(x):
    return np.sum(x**2)

init_x = np.array([-3.0, 4.0])
a = gradient_descent(function_2, init_x, 0.1, 100)

print(a)
