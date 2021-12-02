import pytest

import numpy as np
from numpy.testing import assert_array_equal

from pydeco.problem.centralized_lq import CeLQ


@pytest.fixture()
def setup():
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


def test_celq(setup):
    A, B, Q, R = setup

    n_agents = 3
    edges = [(1, 2), (2, 1), (2, 3), (3, 2)]
    neighbors = {0: [1, ], 1: [0, 2], 2: [1, ]}

    celq = CeLQ(n_agents, edges, A, B, Q, R)

    # model params
    A1, B1, Q1, R1 = celq.get_model()
    A2, B2, Q2, R2 = convert_to_ma_model(A, B, Q, R, n_agents, edges)
    assert_array_equal(A1, A2)
    assert_array_equal(B1, B2)
    assert_array_equal(Q1, Q2)
    assert_array_equal(R1, R2)

    # check reward
    initial_state = np.array([
        1, 0, 0, 0, 0, 5, 0, 0, 0, 0, 10, 0, 0, 0, 0,
    ])
    s0 = celq.reset(initial_state)

    action = np.ones((3 * 3, 1))

    r = 0.
    for i in range(n_agents):
        s0_i = s0[i * 5:i * 5 + 5]
        a0_i = action[i * 3:i * 3 + 3]

        part1 = s0_i.T @ Q @ s0_i + a0_i.T @ R @ a0_i
        r += part1.item()

        for j in neighbors[i]:
            s0_j = s0[j * 5:j * 5 + 5]
            part2 = (s0_i - s0_j).T @ Q @ (s0_i - s0_j)
            r += part2.item()

    curr_reward, next_state = celq.step(action)

    assert curr_reward == r


def convert_to_ma_model(A, B, Q, R, n_agents, edges):
    I_n = np.eye(n_agents)
    A_ = np.kron(I_n, A)
    B_ = np.kron(I_n, B)
    R_ = np.kron(I_n, R)

    q_m, q_n = Q.shape
    Q_ = np.zeros((n_agents * q_m, n_agents * q_n))

    # hardcoded links
    # diagonal
    Q_[:5, :5] = 2 * Q
    Q_[5:10, 5:10] = 3 * Q
    Q_[10:15, 10:15] = 2 * Q

    # off-diagonal
    Q_[5:10, :5] = -Q
    Q_[5:10, 10:15] = -Q
    Q_[:5, 5:10] = -Q
    Q_[10:15, 5:10] = -Q

    return A_, B_, Q_, R_
