import pytest

import numpy as np

from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ
from pydeco.constants import TrainMethod, PolicyEvaluation

from tests.cases import goerges19
from tests.utils import true_P_K


def test_fit_iterative(goerges19):
    A, B, Q, R, s0, K0 = goerges19

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 0.9
    lqr.train(
        lq,
        TrainMethod.ITERATIVE,
        gamma=gamma,
        max_iter=100,
    )

    # comparison
    true_P, true_K = true_P_K(A, B, Q, R, gamma)

    np.testing.assert_array_almost_equal(lqr.P, true_P, decimal=7)
    np.testing.assert_array_almost_equal(lqr.K, true_K, decimal=7)

    assert lqr._calibrated


def test_fit_dare(goerges19):
    A, B, Q, R, s0, K0 = goerges19

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 0.9
    lqr.train(
        lq,
        TrainMethod.DARE,
        gamma=gamma,
    )

    # comparison
    true_P, true_K = true_P_K(A, B, Q, R, gamma)

    np.testing.assert_array_almost_equal(lqr.P, true_P, decimal=7)
    np.testing.assert_array_almost_equal(lqr.K, true_K, decimal=7)

    assert lqr._calibrated


def test_simulate_trajectory(goerges19):
    A, B, Q, R, s0, K0 = goerges19

    # sim params
    s0 = np.array([1, 1, 1])
    t0, tn, n_steps = 0., 1., 20

    lq = LQ(A, B, Q, R)

    lqr = LQR()

    lqr.train(
        lq,
        TrainMethod.ITERATIVE,
    )

    xs, us, tcost = lqr.simulate_trajectory(lq, s0, t0, tn, n_steps)

    assert -4.57543253638 == pytest.approx(tcost, abs=1e-8)
