import numpy as np

if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]])
    n_l = 3
    I_n = np.eye(n_l)

    A_ = np.kron(I_n, A)

    b = 1

    print(b)
    print(A_)
