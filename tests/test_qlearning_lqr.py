import pytest

import numpy as np

from pydeco.controller.qlearning_lqr import QlearningLQR
from pydeco.problem.lq import LQ


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


def test_build_feature_vector():
    lqr = QlearningLQR()

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
    lqr = QlearningLQR()

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
