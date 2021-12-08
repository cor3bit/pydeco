import pytest

import numpy as np


@pytest.fixture()
def simple():
    n_x = 2
    n_u = 1

    # env params
    A = np.array([
        [1., 2., ],
        [2., 1., ],
    ])

    B = np.eye(n_u)
    Q = -np.eye(n_x)
    R = -np.eye(n_u)

    K0 = -np.ones((n_x, n_x))
    s0 = np.zeros(shape=(n_x,))

    return A, B, Q, R, s0, K0


@pytest.fixture()
def wang20():
    n_x = 4
    n_u = 4

    # env params
    A = np.array([
        [0.2, 0.4, 0.1, 0.01],
        [0.4, 0.2, 0.3, 0.1],
        [0.1, 0.3, 0.3, 0.4],
        [0.2, 0.1, 0.5, 0.3],
    ])

    B = np.eye(n_u)
    Q = -np.eye(n_x)
    R = -np.eye(n_u)

    K0 = -np.array([
        [1, 1, 0.0004, 2],
        [1, 0.2, 1, 0.1],
        [4, 0.1, 1, 3],
        [0.2, 0.1, 0.3, 0.2],
    ])
    s0 = np.full(shape=(n_x,), fill_value=0.01)

    return A, B, Q, R, s0, K0


@pytest.fixture()
def goerges19():
    n_x = 3
    n_u = 3

    # env params
    A = np.array([
        [0.3, 0.7, 0],
        [0.4, 0.5, 0.2],
        [0, 0.2, 0.4],
    ])
    B = np.eye(n_u)

    Q = -np.eye(n_x)
    R = -np.eye(n_u)

    K0 = np.full(shape=(n_x, n_x), fill_value=-.01)
    s0 = np.full(shape=(n_x,), fill_value=0.)

    return A, B, Q, R, s0, K0


@pytest.fixture()
def alemzadeh18():
    # UAV example
    n_s = 5
    n_a = 3

    A = np.array(
        [
            [0.0000, 0.0000, 1.1320, 0.0000, -1.000],
            [0.0000, -0.0538, -0.1712, 0.0000, 0.0705],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0485, 0.0000, -0.8556, -1.013],
            [0.0000, -0.2909, 0.0000, 1.0532, -0.6859],
        ]
    )

    B = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [-0.120, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [4.4190, 0.0000, -1.665],
            [1.5750, 0.0000, -0.0732],
        ]
    )

    Q = -np.eye(n_s)
    R = -np.eye(n_a)

    return A, B, Q, R
