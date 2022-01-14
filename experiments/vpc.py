import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pydeco.constants import *
from pydeco.problem.platoon import EVPlatoon
from pydeco.controller.dlqr import LocalLQR, MultiAgentLQR
from pydeco.viz.plot2d import plot_evolution


def run_experiment():
    # identical systems
    n_a = 2
    n_s = 1

    n_agents = 3
    communication_map = {
        0: [1, ],
        1: [0, 2, ],
        2: [1, ],
    }

    communication_links = [(agent_id, ng) for agent_id, ngs in communication_map.items() for ng in ngs]

    # training params
    s0_0 = np.array([0.1, 0.1, 0, 0, 0.1])
    s0_1 = np.array([0.2, 0.1, 0, 0, 0.1])
    s0_2 = np.array([0.5, 0.1, 0, 0, 0.1])
    s0 = np.concatenate((s0_0, s0_1, s0_2))

    n_steps = 20
    ts = np.linspace(0, 1, num=n_steps + 1)

    # -------------- solve DISTRIBUTED --------------
    ma_env = EVPlatoon(
        n_agents, communication_map, Q, R)

    ma_lqr = MultiAgentLQR(n_agents)

    # training params
    initial_states = [s0_0, s0_1, s0_2]
    gamma = 1.0
    eps = 1e-7
    max_policy_evals = 700
    max_policy_improves = 150
    reset_every_n = 1

    # TODO try other initial policies

    sa_k_star = -np.full((3, 5), fill_value=1)

    # initial_policies = [sa_k_star, sa_k_star, sa_k_star]

    ma_lqr.train(
        ma_env,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN_RLS,
        gamma=gamma,
        eps=eps,
        alpha=0.1,
        max_policy_evals=max_policy_evals,
        max_policy_improves=max_policy_improves,
        reset_every_n=reset_every_n,
        initial_states=initial_states,
        initial_state_fn=initial_state_fn,
        sa_initial_policy=sa_k_star,
        optimal_controller=None,
    )

    # plotting
    xs_star, us_star, tcost = ma_lqr.simulate_trajectory(
        ma_env, initial_states, 0, 1, n_steps=n_steps,
    )
    plot_evolution(xs_star, ts, [0, 5, 10], 'TEST')


def initial_state_fn():
    # height error relative to ground or guidance aid, in m;
    x1 = np.random.exponential(scale=1., size=1).item()

    # forward speed, in m/sec;
    x2 = np.random.exponential(scale=0.1, size=1).item()

    # pitch angle, in degrees;
    x3 = np.random.exponential(scale=0.1, size=1).item()

    # rate of change of pitch angle, in degree/sec;
    x4 = np.random.exponential(scale=0.1, size=1).item()

    # vertical speed, in m/sec
    x5 = np.random.exponential(scale=0.1, size=1).item()

    return np.array([x1, x2, x3, x4, x5, ])


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.random.seed(42)
    sns.set_style('whitegrid')

    run_experiment()
