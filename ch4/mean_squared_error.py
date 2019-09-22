import numpy as np

def mean_squared_error(y, t):
    print(0.5 * np.sum((y-t)**2))
    # return 0.5 * np.sum((y-t)**2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [.1, .05, .6, .0, .05, .1, .0, .1, .0, .0]
mean_squared_error(np.array(y), np.array(t))

y = [.1, .05, .1, .0, .05, .1, .0, .6, .0, .0]
mean_squared_error(np.array(y), np.array(t))
