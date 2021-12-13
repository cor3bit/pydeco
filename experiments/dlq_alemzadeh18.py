import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pydeco.constants import *
from pydeco.problem.centralized_lq import CeLQ
from pydeco.problem.distributed_lq import DiLQ, create_distributed_envs
from pydeco.controller.lqr import LQR
from pydeco.controller.dlqr import LocalLQR
from pydeco.viz.plot2d import plot_evolution


def run_experiment():
    # identical systems
    n_s = 5
    n_a = 3
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

    communication_links = []
    for k, v in communication_map.items():
        for ng in v:
            communication_links.append((k, ng))

    double_count_rewards = False

    coupled_dynamics = False
    coupled_rewards = True

    # solve CENTRALIZED
    # Q-learning
    print('\nCeLQ:')

    lq = CeLQ(n_agents, communication_links, double_count_rewards, A, B, Q, R)

    print('\nRiccati LQR:')
    s0_0 = np.array([0.1, 5, 0, 0, 5])
    s0_1 = np.array([0.2, 1, 0, 0, 1])
    s0_2 = np.array([0.5, 1, 0, 0, 1])
    s0 = np.concatenate((s0_0, s0_1, s0_2))
    lqr = LQR()
    lqr.train(lq, method=TrainMethod.ANALYTICAL, initial_state=s0)
    P_star = lqr.P
    K_star = lqr.K
    # print(f'P: {P_star}')
    print(f'K: {K_star}')

    xs_star, us_star, tcost = lqr.simulate_trajectory(lq, s0, 0, 1, n_steps=10)
    ts = np.linspace(0, 1, num=10 + 1)
    plot_evolution(xs_star, ts, [0, 5, 10], 'TEST')



    # solve DISTRIBUTED

    # create envs
    envs = create_distributed_envs(
        n_agents, communication_map, coupled_dynamics, coupled_rewards, A, B, Q, R)

    # initial state
    initial_states = [s0_0, s0_1, s0_2]
    # initial_states = [np.zeros((n_s, 1))]

    curr_states = {
        i: env.reset(initial_state) for i, (env, initial_state) in enumerate(zip(envs, initial_states))
    }

    agents = [LocalLQR() for _ in range(n_agents)]

    # initialize
    for env, agent in zip(envs, agents):
        agent.initialize_qlearn_ls(env.n_s, env.n_a)

    # training params
    max_policy_improves = 10
    max_policy_evals = 400
    gamma = 1.

    all_converged = False
    iter = 0

    # policy improvement loop
    pbar = tqdm(total=max_policy_improves * max_policy_evals)

    while not all_converged and iter < max_policy_improves:
        # reset covar
        for agent in agents:
            agent.reset_covar()

        # policy evaluation loop
        for _ in range(max_policy_evals):

            next_states = []
            for agent_id, (agent, env) in enumerate(zip(agents, envs)):
                next_state = agent.eval_policy(env, agent_id, curr_states, gamma)
                next_states.append(next_state)

            curr_states = {
                i: next_state for i, next_state in enumerate(next_states)
            }

            pbar.update(1)

        # policy improvement step
        all_converged = True
        for agent, env in zip(agents, envs):
            converged = agent.improve_policy()
            all_converged &= converged

        # update iteration
        iter += 1

    pbar.close()

    # PLOTTING
    xs_star, us_star, tcost = simulate_dilq_trajectories(envs, agents, initial_states, 0, 1, n_steps=10)
    plot_evolution(xs_star, ts, [0, 5, 10], 'TEST')


def simulate_dilq_trajectories(envs, agents, initial_states, t0, tn, n_steps):

    curr_states = {
        i: env.reset(initial_state) for i, (env, initial_state) in enumerate(zip(envs, initial_states))
    }

    # calculate optimal controls
    time_grid = np.linspace(t0, tn, num=n_steps + 1)
    x_k = env.get_state()
    xs = [x_k]
    us = []

    for env, agent in zip(envs, agents):

        total_cost = .0
        for k in time_grid[:-1]:
            # optimal control at k
            u_k = agent.act(x_k)
            us.append(u_k)

            # update state
            r_k, next_x_k = env.step(u_k)
            xs.append(x_k)

            # increment stage cost
            total_cost += r_k

            # update state
            x_k = next_x_k

        rf = env.terminal_cost(x_k)
        total_cost += rf

        xs = np.squeeze(np.stack(xs))
        us = np.squeeze(np.stack(us))

    return xs, us, total_cost



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
