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

    celq = CeLQ(n_agents, communication_links, double_count_rewards, A, B, Q, R)

    celqr = LQR()

    celqr.train(
        celq,
        method=TrainMethod.DARE,
        initial_state=s0,
    )
    P_star_celq = celqr.P
    K_star_celq = celqr.K
    print(f'P: {P_star_celq}')
    print(f'K: {K_star_celq}')

    # plotting
    n_steps = 5
    ts = np.linspace(0, 1, num=n_steps + 1)
    xs_star, us_star, tcost = celqr.simulate_trajectory(celq, s0, 0, 1, n_steps=n_steps)
    plot_evolution(xs_star, ts, [0, 1, 2], 'TEST')

    # -------------- solve DISTRIBUTED --------------
    print('\nD-LQ:')

    ma_env = MultiAgentLQ(n_agents, communication_map, coupled_dynamics,
                          coupled_rewards, A, B, Q, R)

    ma_lqr = MultiAgentLQR(n_agents)

    # training params
    initial_states = [s0_0, s0_1, s0_2]
    gamma = 1.0
    eps = 1e-6
    max_policy_evals = 200
    max_policy_improves = 20
    reset_every_n = 5

    sa_k_star = np.full((1, 1), fill_value=-.01)

    # initial_policies = [sa_k_star, sa_k_star, sa_k_star]

    # ma_lqr.train(
    #     ma_env,
    #     method=TrainMethod.DARE,
    #     gamma=gamma,
    # )

    ma_lqr.train(
        ma_env,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN_GN,
        gamma=gamma,
        eps=eps,
        alpha=0.1,
        max_policy_evals=max_policy_evals,
        max_policy_improves=max_policy_improves,
        reset_every_n=reset_every_n,
        initial_states=initial_states,
        sa_initial_policy=sa_k_star,
        optimal_controller=K_star_celq,
    )

    # for agent in ma_lqr._agent_map.values():
    #     print(agent.K)
    K_sim = ma_lqr._reconstruct_full_K(ma_env)
    print(f'K_sim: {K_sim}')

    # plotting
    xs_sim, us_sim, tcost_sim = ma_lqr.simulate_trajectory(
        ma_env, initial_states, 0, 1, n_steps=n_steps,
    )
    plot_evolution(xs_sim, ts, [0, 1, 2], 'TEST')

    # ------ get centralized P from centralized K ------
    # lqr2 = LQR()
    #
    # lqr2._init_gpi(lq, s0, K_sim)
    #
    # lqr2._policy_eval_qlearn_rls(
    #     lq,
    #     gamma,
    #     eps,
    #     max_policy_evals=5000,
    #     reset_every_n=reset_every_n,
    #     gpi_iter=0,
    #     initial_state=s0,
    # )
    # print(f'P_sim: {lqr2.P}')


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
