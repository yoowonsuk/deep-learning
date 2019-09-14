import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def step_function(x):
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5., 5., .1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-.1, 1.1)
plt.show()
