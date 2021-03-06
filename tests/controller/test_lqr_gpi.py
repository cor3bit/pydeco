import pytest

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ
from pydeco.constants import *

from tests.cases import goerges19
from tests.utils import true_P_K


def test_fit_qlearn_rls(goerges19):
    np.random.seed(42)

    A, B, Q, R, s0, K0 = goerges19

    # pydeco solution
    # env
    lq = LQ(A, B, Q, R)

    # agent
    lqr = LQR()

    # train params
    gamma = 1.0

    n_evals = 100
    n_improves = 5
    reset_every_n = 1000

    lqr.train(
        lq,
        TrainMethod.GPI,
        PolicyEvaluation.QLEARN_RLS,
        gamma=gamma,

        max_policy_evals=n_evals,
        max_policy_improves=n_improves,
        reset_every_n=reset_every_n,

        initial_state=s0,
        initial_policy=K0,
    )

    calc_P, calc_K = lqr.P, lqr.K

    true_P, true_K = true_P_K(A, B, Q, R, gamma)

    max_diff = np.max(np.abs(calc_K - true_K))

    print(max_diff)
    assert max_diff < .0001

    # np.testing.assert_array_almost_equal(calc_P, true_P, decimal=7)
    # np.testing.assert_array_almost_equal(calc_K, true_K, decimal=7)


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
    alpha = 0.01
    eps = 1e-8

    n_evals = 200
    n_improves = 30
    reset_every_n = 5

    lqr.train(
        lq,
        TrainMethod.GPI,
        PolicyEvaluation.QLEARN,
        gamma=gamma,

        initial_state=s0,
        initial_policy=K0,

        max_policy_evals=n_evals,
        max_policy_improves=n_improves,
        reset_every_n=reset_every_n,

        alpha=alpha,
        eps=eps,
    )

    calc_P, calc_K = lqr.P, lqr.K

    true_P, true_K = true_P_K(A, B, Q, R, gamma)

    max_diff = np.max(np.abs(calc_K - true_K))

    print(max_diff)
    assert max_diff < .01

    # np.testing.assert_array_almost_equal(calc_P, true_P, decimal=7)
    # np.testing.assert_array_almost_equal(calc_K, true_K, decimal=7)


def test_build_feature_vector():
    lqr = LQR()

    s = np.array([1, 2])
    a = np.array([3])
    p = 6

    x = lqr._build_feature_vector(p, s, a)

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
    x = lqr._build_feature_vector(p, s, a)
    theta = np.arange(p).reshape((p, 1))
    v2 = x.T @ theta
    v2 = v2.reshape((-1,))[0]

    # value based on (s,a)
    sa = np.concatenate((s, a))
    H = lqr._convert_to_parameter_matrix(theta, n_q)
    v1 = sa.T @ H @ sa

    assert v1 == v2
