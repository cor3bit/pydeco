import pytest

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pydeco.controller.analytic_lqr import AnalyticalLQR
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


def test_fit(setup):
    A, B, Q, R = setup

    # numpy solution
    np_P = dare(
        np.array(A),
        np.array(B),
        np.array(Q),
        np.array(R),
    )

    # pydeco solution
    lqr = AnalyticalLQR()
    lq = LQ(A, B, Q, R)
    lqr.train(lq, n_steps=10)

    # comparison
    # P
    np.testing.assert_array_almost_equal(
        np_P,
        np.array(lqr._P),
        decimal=7,
    )

    # K
    np.testing.assert_array_almost_equal(
        np.array([[-0.17794839, -0.39653113, -0.01332207],
                  [-0.2498468, -0.33199662, -0.12628551],
                  [-0.01135366, -0.12226867, -0.21419297]]),
        np.array(lqr._K),
        decimal=7,
    )


def test_simulate_trajectory(setup):
    A, B, Q, R = setup

    # sim params
    x0 = np.array([1, 1, 1])
    t0, tn, n_steps = 0., 1., 20

    lq = LQ(A, B, Q, R)
    lqr = AnalyticalLQR()
    lqr.train(lq, n_steps=n_steps)
    xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)

    assert 4.57543253638 == pytest.approx(tcost, abs=1e-8)
