import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pydeco.problem.distributed_lq import *

from tests.cases import wang20


def test_local_lq(wang20):
    A, B, Q, R, s0, K0 = wang20

    n_agents = 3
    communication_map = {0: [1, ], 1: [0, 2], 2: [1, ]}

    # envs = create_distributed_envs(n_agents, communication_map, False, True, A, B, Q, R)

    agent_id = 1

    lolq = LocalLQ(
        communication_map[agent_id],
        coupled_dynamics=False,
        coupled_rewards=True,
        system_matrix=A,
        control_matrix=B,
        state_reward_matrix=Q,
        action_reward_matrix=R,
    )

    n_s, n_a = B.shape

    s0_i = np.reshape(s0 * agent_id, (-1, 1))
    a0_i = np.ones(n_a).reshape((-1, 1))
    z0_i = [np.reshape(s0 * j, (-1, 1)) for j in communication_map[agent_id]]

    lolq.reset(s0_i)
    curr_reward, next_state = lolq.step(a0_i, information=z0_i)

    # next state
    s1_i = A @ s0_i + B @ a0_i
    assert_array_equal(s1_i, next_state)

    # reward
    r0_i = _calc_coupled_reward(agent_id, communication_map[agent_id], Q, R, s0, n_a)
    assert r0_i == curr_reward


def _calc_coupled_reward(i, neighbors, Q, R, s0, n_a):
    r = 0.

    # self
    s0_i = np.reshape(s0 * i, (-1, 1))
    a0_i = np.ones(n_a).reshape((-1, 1))

    part1 = s0_i.T @ Q @ s0_i + a0_i.T @ R @ a0_i
    r += part1.item()

    # others
    for j in neighbors:
        s0_j = np.reshape(s0 * j, (-1, 1))
        part2 = (s0_i - s0_j).T @ Q @ (s0_i - s0_j)
        r += part2.item()

    return r
