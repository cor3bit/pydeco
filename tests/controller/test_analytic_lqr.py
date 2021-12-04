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


def test_fit(setup):
    A, B, Q, R = setup

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 1.0
    lqr.train(lq, TrainMethod.ANALYTICAL, gamma)

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


def test_simulate_trajectory(setup):
    A, B, Q, R = setup

    # sim params
    s0 = np.array([1, 1, 1])
    t0, tn, n_steps = 0., 1., 20

    lq = LQ(A, B, Q, R)

    lqr = LQR()
    lqr.train(lq, TrainMethod.ANALYTICAL)

    xs, us, tcost = lqr.simulate_trajectory(lq, s0, t0, tn, n_steps)

    assert 4.57543253638 == pytest.approx(tcost, abs=1e-8)
