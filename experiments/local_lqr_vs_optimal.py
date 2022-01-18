import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pydeco.constants import *
from pydeco.problem.lq import LQ
from pydeco.problem.centralized_lq import CeLQ
from pydeco.problem.distributed_lq import LocalLQ, MultiAgentLQ
from pydeco.controller.lqr import LQR
from pydeco.controller.dlqr import LocalLQR, MultiAgentLQR
from pydeco.viz.plot2d import plot_evolution


def run_experiment():
    # identical systems
    A, B, Q, R = problem_setup()

    n_agents = 3
    communication_map = {
        0: [1, ],
        1: [0, 2, ],
        2: [1, ],
    }

    communication_links = [(agent_id, ng) for agent_id, ngs in communication_map.items() for ng in ngs]

    double_count_rewards = False

    coupled_dynamics = False
    coupled_rewards = True

    # training params
    s0_0 = np.array([0.1, 0.1, 0, 0, 0.1])
    s0_1 = np.array([0.2, 0.1, 0, 0, 0.1])
    s0_2 = np.array([0.5, 0.1, 0, 0, 0.1])
    s0 = np.concatenate((s0_0, s0_1, s0_2))

    # -------------- solve CENTRALIZED --------------
    # print('\nCeLQ:')
    #
    # lq = CeLQ(n_agents, communication_links, double_count_rewards, A, B, Q, R)
    #
    # lqr = LQR()
    #
    # lqr.train(
    #     lq,
    #     method=TrainMethod.ANALYTICAL,
    #     initial_state=s0,
    # )
    # # P_star = lqr.P
    # K_star = lqr.K
    # # print(f'P: {P_star}')
    # print(f'K: {K_star}')
    #
    # # plotting
    # xs_star, us_star, tcost = lqr.simulate_trajectory(lq, s0, 0, 1, n_steps=n_steps)
    # plot_evolution(xs_star, ts, [0, 5, 10], 'TEST')

    local_lq = LocalLQ(
        communication_map[0], coupled_dynamics, coupled_rewards,
        A, B, Q, R,
    )

    # lq_generic = LQ(*local_lq.get_full_model(), check_dimensions=False)

    lqr = LQR()

    lqr.train(local_lq, method=TrainMethod.ITERATIVE, max_iter=100)
    # print(f'P: {lqr.P}')
    # print(f'K: {lqr.K}')

    K = lqr.K
    a = 1

    # n_steps = 20
    # ts = np.linspace(0, 1, num=n_steps + 1)
    #
    # xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    # print(f'Total Cost: {tcost}')


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
