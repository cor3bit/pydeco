import pytest

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ
from pydeco.constants import TrainMethod


@pytest.fixture()
def setup():
    # DS params
    A = np.array([
        [0.3, 0.7, 0],
        [0.4, 0.5, 0.2],
        [0, 0.2, 0.4],
    ])
    B = np.eye(3)
    Q = np.eye(3)
    R = np.eye(3)

    return A, B, Q, R


def test_fit_qlearn(setup):
    A, B, Q, R = setup

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 1.0
    lqr.train(lq, TrainMethod.QLEARN, gamma)

    # comparison
    # numpy solution
    np_P = dare(
        np.array(A * np.sqrt(gamma)),
        np.array(B * np.sqrt(gamma)),
        np.array(Q),
        np.array(R),
    )

    # P
    np.testing.assert_array_almost_equal(
        lqr.P,
        np.array(lqr._P),
        decimal=7,
    )

    # K
    np.testing.assert_array_almost_equal(
        np.array([[-0.17794839, -0.39653113, -0.01332207],
                  [-0.2498468, -0.33199662, -0.12628551],
                  [-0.01135366, -0.12226867, -0.21419297]]),
        np.array(lqr.K),
        decimal=7,
    )


def test_fit_qlearn_ls(setup):
    A, B, Q, R = setup

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 1.0
    lqr.train(lq, TrainMethod.QLEARN_LS, gamma)

    # comparison
    # numpy solution
    np_P = dare(
        np.array(A * np.sqrt(gamma)),
        np.array(B * np.sqrt(gamma)),
        np.array(Q),
        np.array(R),
    )

    # P
    np.testing.assert_array_almost_equal(
        lqr.P,
        np.array(lqr._P),
        decimal=7,
    )

    # K
    np.testing.assert_array_almost_equal(
        np.array([[-0.17794839, -0.39653113, -0.01332207],
                  [-0.2498468, -0.33199662, -0.12628551],
                  [-0.01135366, -0.12226867, -0.21419297]]),
        np.array(lqr.K),
        decimal=7,
    )


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
