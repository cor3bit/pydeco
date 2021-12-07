import numpy as np
from time import perf_counter

# s, a -> theta
n_q = 100
sa = np.arange(n_q)
n = sa.shape[0]
p = int(n_q * (n_q + 1) / 2)

# TODO optimize building quadratic features
t1 = perf_counter()
# y = np.multiply(sa.reshape((-1, 1)), sa.reshape((1, -1)))
y = np.outer(sa, sa)
y_ind = np.triu_indices(y.shape[0])
y_ = y[y_ind].reshape((-1, 1))
tf_1 = perf_counter() - t1
print(tf_1)


t2 = perf_counter()
x = np.empty((p, 1))
c = 0
for i in range(n):
    for j in range(i, n):
        x[c, 0] = sa[i] * sa[j]
        c += 1

tf_2 = perf_counter() - t2

print(tf_2)
