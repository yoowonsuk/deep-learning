import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def function_2(x):
    return x[0]**2 + x[1]**2 # return np.sum(x**2), and consider x numpy array

X = np.arange(-3, 3, 0.25)
Y = np.arange(-3, 3, 0.25)
X, Y = np.meshgrid(X, Y) # create the base grid
Z = X**2 + Y**2

fig = plt.figure() # create new figure
ax =fig.gca(projection='3d') # or ax = fig.add_subplot(111, projection='3d'), create Axes3D object
surf = ax.plot_surface(X, Y, Z)

'''
# line/scatter plot : x,y,z는 1차원 배열
ax.plot(x,y,z,...)
ax.scatter(x,y,z,...)

# surface, wireframe plot : X,Y,Z는 2차원 배열
ax.plot_surface(X,Y,Z,...)
ax.plot_wireframe(X,Y,Z,...)

# bar3d plot : x, y, z는 1차원 배열
ax.bar3d(x,y,width,depth,z,...)
'''

plt.show()
