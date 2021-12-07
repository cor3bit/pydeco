import numpy as np
from scipy.linalg import solve_discrete_are as dare


def true_P_K(A, B, Q, R, gamma):
    # numpy solution
    P = dare(
        np.array(A * np.sqrt(gamma)),
        np.array(B * np.sqrt(gamma)),
        np.array(Q),
        np.array(R),
    )

    K = - np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    return P, K
