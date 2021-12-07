import pytest

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ
from pydeco.constants import TrainMethod


@pytest.fixture()
def wang20():
    # DS params: wang20
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

    K0 = -1. * np.array([
        [1, 1, 0.0004, 2],
        [1, 0.2, 1, 0.1],
        [4, 0.1, 1, 3],
        [0.2, 0.1, 0.3, 0.2],
    ])
    s0 = np.full(shape=(4,), fill_value=0.01)

    return A, B, Q, R, s0, K0


@pytest.fixture()
def goerges19():
    # DS params: wang20
    n_x = 3
    n_u = 3

    # env params
    A = np.array([
        [0.3, 0.7, 0],
        [0.4, 0.5, 0.2],
        [0, 0.2, 0.4],
    ])
    B = np.eye(3)
    Q = -np.eye(3)
    R = -np.eye(3)

    K0 = np.full(shape=(n_x, n_x), fill_value=-1.)
    s0 = np.full(shape=(n_x,), fill_value=1.)

    return A, B, Q, R, s0, K0


def test_fit_qlearn_ls(goerges19):
    np.random.seed(42)

    A, B, Q, R, s0, K0 = goerges19

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 1.0

    lqr.train(lq, TrainMethod.QLEARN_LS, gamma, initial_state=s0, initial_policy=K0)

    calc_P, calc_K = lqr.P, lqr.K

    true_P, true_K = true_P_K(A, B, Q, R, gamma)

    aaa = 111

    # np.testing.assert_array_almost_equal(calc_P, true_P, decimal=7)
    np.testing.assert_array_almost_equal(calc_K, true_K, decimal=7)


def test_fit_qlearn(goerges19):
    np.random.seed(42)

    A, B, Q, R, s0, K0 = goerges19

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 1.0
    alpha = 0.0001
    max_iter = 50000

    lqr.train(lq, TrainMethod.QLEARN, gamma, initial_state=s0,
              initial_policy=K0, alpha=alpha, max_iter=max_iter)

    calc_P, calc_K = lqr.P, lqr.K

    true_P, true_K = true_P_K(A, B, Q, R, gamma)

    # np.testing.assert_array_almost_equal(calc_P, true_P, decimal=7)
    np.testing.assert_array_almost_equal(calc_K, true_K, decimal=7)


def test_build_feature_vector():
    lqr = LQR()

    s = np.array([1, 2])
    a = np.array([3])
    p = 6

    x = lqr._build_feature_vector(s, a, p)

    np.testing.assert_array_almost_equal(
        np.array([[1.], [2.], [3.], [4.], [6.], [9.]]),
        x,
        decimal=7,
    )


def test_convert_to_parameter_matrix():
    lqr = LQR()

    # set-up
    s = np.array([1, 2])
    a = np.array([3])
    p = 6
    n_q = 3

    # value based on quadratic features
    x = lqr._build_feature_vector(s, a, p)
    theta = np.arange(p).reshape((p, 1))
    v2 = x.T @ theta
    v2 = v2.reshape((-1,))[0]

    # value based on (s,a)
    sa = np.concatenate((s, a))
    H = lqr._convert_to_parameter_matrix(theta, n_q)
    v1 = sa.T @ H @ sa

    assert v1 == v2


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
