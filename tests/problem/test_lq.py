import pytest

import numpy as np
from numpy.testing import assert_array_equal

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
    Q = -np.eye(3)
    R = -np.eye(3)

    return A, B, Q, R


def test_lq(setup):
    A, B, Q, R = setup

    # init
    lq = LQ(A, B, Q, R)
    assert lq.n_s == 3
    assert lq.n_a == 3

    # reset
    s0 = np.ones(shape=(3,))
    lq.reset(s0)
    curr_state = lq.get_state()
    assert_array_equal(curr_state, s0.reshape((3, 1)))

    # step
    curr_action = np.ones(shape=(3, 1))
    curr_reward, next_state = lq.step(curr_action)

    s1 = A @ s0.reshape((3, 1)) + B @ curr_action
    assert_array_equal(next_state, s1)
    assert_array_equal(lq.get_state(), s1)

    r0 = s0 @ Q @ s0 + curr_action.T @ R @ curr_action
    assert curr_reward == r0[0][0]

    # terminal reward
    final_reward = lq.terminal_cost()
    rf = s1.T @ Q @ s1
    assert final_reward == rf[0][0]
