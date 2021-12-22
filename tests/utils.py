import numpy as np
from scipy.linalg import solve_discrete_are as dare


def true_P_K(A, B, Q, R, gamma):
    A_adj = A * np.sqrt(gamma)
    B_adj = B * np.sqrt(gamma)

    # numpy solution
    P = dare(
        np.array(A_adj),
        np.array(B_adj),
        np.array(Q),
        np.array(R),
    )

    K = - np.linalg.inv(R + B_adj.T @ P @ B_adj) @ B_adj.T @ P @ A_adj

    return P, K
