import numpy as np

x = ([1, 2], [3, 4])

print(np.max(x, axis=1)) # 1차원 배열 -> 뺄셈이 이상해진다.

x = x - np.max(x, axis = 1)

print(x)



y = ([0, 0, 0, 1, 0])

# y = y.argmax(axis=1) # 공통 적용 문제

print(y)

x = np.array([1, -1, 3])

grad = np.zeros_like(x)
grad[x>=0] = 1
print(grad)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

p = (1.0 - sigmoid(x)) * sigmoid(x)

print(p)
