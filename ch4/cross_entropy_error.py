import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    print(-np.sum(t*np.log(y+delta)))
    # return -np.sum(t*np.log(y+delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [.1, .05, .6, .0, .05, .1, .0, .1, .0, .0]
cross_entropy_error(np.array(y), np.array(t))

y = [.1, .05, .1, .0, .05, .1, .0, .6, .0, .0]
cross_entropy_error(np.array(y), np.array(t))
