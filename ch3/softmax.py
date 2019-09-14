import numpy as np

a = np.array([.3, 2.9, 4.])

exp_a = np.exp(a)

print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([.3, 2.9, 4.])
y = softmax(a)

print(np.sum(y))
