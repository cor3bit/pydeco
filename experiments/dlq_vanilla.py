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
    s0_0 = np.array([0.1])
    s0_1 = np.array([0.2])
    s0_2 = np.array([0.5])
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
    P_star = lqr.P
    K_star = lqr.K
    print(f'P: {P_star}')
    print(f'K: {K_star}')

    # plotting
    n_steps = 5
    ts = np.linspace(0, 1, num=n_steps + 1)
    xs_star, us_star, tcost = lqr.simulate_trajectory(lq, s0, 0, 1, n_steps=n_steps)
    plot_evolution(xs_star, ts, [0, 1, 2], 'TEST')

    # -------------- solve DISTRIBUTED --------------
    ma_env = MultiAgentLQ(
        n_agents, communication_map, coupled_dynamics, coupled_rewards, A, B, Q, R)

    ma_lqr = MultiAgentLQR(n_agents, optimal_controller=K_star)

    # training params
    initial_states = [s0_0, s0_1, s0_2]
    gamma = 1.0
    eps = 1e-6
    max_policy_evals = 100
    max_policy_improves = 20
    reset_every_n = 100

    sa_k_star = np.full((1, 1), fill_value=-.01)

    # initial_policies = [sa_k_star, sa_k_star, sa_k_star]

    ma_lqr.train(
        ma_env,
        gamma,
        eps,
        max_policy_evals,
        max_policy_improves,
        reset_every_n,
        initial_states,
        sa_k_star,
    )

    # for agent in ma_lqr._agent_map.values():
    #     print(agent.K)
    K_sim = ma_lqr._reconstruct_full_K(ma_env)
    print(K_sim)

    # plotting
    xs_sim, us_sim, tcost_sim = ma_lqr.simulate_trajectory(
        ma_env, initial_states, 0, 1, n_steps=n_steps,
    )
    plot_evolution(xs_sim, ts, [0, 1, 2], 'TEST')

    # ------ initialize LQR from policy
    lqr2 = LQR()
    lqr2.K = K_sim

    lqr2._qlearn_rls_policy_eval(
        lq,
        max_policy_evals=5000,
        reset_every_n=reset_every_n,
        initial_state=s0,
        initial_policy=K_sim,
    )
    print(f'P_sim: {lqr2.P}')


def problem_setup():
    n_s = 1
    n_a = 1

    A = np.array([[0.2]])
    B = np.array([[0.1]])
    Q = -np.eye(n_s)
    R = -np.eye(n_a)

    return A, B, Q, R


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.random.seed(42)
    sns.set_style('whitegrid')

    run_experiment()
