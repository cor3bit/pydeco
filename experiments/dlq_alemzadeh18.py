import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pydeco.constants import *
from pydeco.problem.centralized_lq import CeLQ
from pydeco.problem.distributed_lq import LocalLQ, MultiAgentLQ
from pydeco.controller.lqr import LQR
from pydeco.controller.dlqr import LocalLQR, MultiAgentLQR
from pydeco.viz.plot2d import plot_evolution


def run_experiment():
    # identical systems
    A, B, Q, R = problem_setup()

    # communication topology
    # n_agents = 6
    # communication_map = {
    #     0: [1, ],
    #     1: [0, 2],
    #     2: [1, 3],
    #     3: [2, 4],
    #     4: [3, 5],
    #     5: [4, ],
    # }

    # n_agents = 2
    # communication_map = {
    #     0: [1, ],
    #     1: [0, ],
    # }

    n_agents = 3
    communication_map = {
        0: [1, ],
        1: [0, 2, ],
        2: [1, ],
    }

    # n_agents = 1
    # communication_map = {
    #     0: [],
    # }

    communication_links = [(agent_id, ng) for agent_id, ngs in communication_map.items() for ng in ngs]

    double_count_rewards = False

    coupled_dynamics = False
    coupled_rewards = True

    # training params
    s0_0 = np.array([0.1, 5, 0, 0, 5])
    s0_1 = np.array([0.2, 1, 0, 0, 1])
    s0_2 = np.array([0.5, 1, 0, 0, 1])
    s0 = np.concatenate((s0_0, s0_1, s0_2))

    # -------------- solve CENTRALIZED --------------
    print('\nCeLQ:')

    lq = CeLQ(n_agents, communication_links, double_count_rewards, A, B, Q, R)

    lqr = LQR()

    lqr.train(
        lq,
        method=TrainMethod.ANALYTICAL,
        initial_state=s0,
    )
    # P_star = lqr.P
    K_star = lqr.K
    # print(f'P: {P_star}')
    print(f'K: {K_star}')

    # plotting
    xs_star, us_star, tcost = lqr.simulate_trajectory(lq, s0, 0, 1, n_steps=10)
    ts = np.linspace(0, 1, num=10 + 1)
    plot_evolution(xs_star, ts, [0, 5, 10], 'TEST')

    # -------------- solve DISTRIBUTED --------------
    ma_env = MultiAgentLQ(
        n_agents, communication_map, coupled_dynamics, coupled_rewards, A, B, Q, R)

    ma_lqr = MultiAgentLQR(n_agents)

    # training params
    initial_states = [s0_0, s0_1, s0_2]
    gamma = 1.0
    eps = 1e-6
    max_policy_evals = 50
    max_policy_improves = 20

    # TODO try other initial policies
    sa_k_star = np.array(
        [[0., 0.06966258, -0.00066418, -0.03478692, 0.29336068, ],
         [0., 0.05054217, 0.08991914, -0.08479701, 0.00070446, ],
         [0., 0.20414527, 0.00281244, -0.60267623, 0.16335665, ], ]
    )

    # sa_k_star = np.full((3, 5), fill_value=-.01)

    # initial_policies = [sa_k_star, sa_k_star, sa_k_star]

    ma_lqr.train(
        ma_env,
        gamma,
        eps,
        max_policy_evals,
        max_policy_improves,
        initial_states,
        sa_k_star,
    )

    # plotting
    # xs_star, us_star, tcost = ma_lqr.simulate_trajectory(
    #     ma_env, initial_states, 0, 1, n_steps=10,
    # )
    # plot_evolution(xs_star, ts, [0, 5, 10], 'TEST')


def problem_setup():
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


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.random.seed(42)
    sns.set_style('whitegrid')

    run_experiment()
