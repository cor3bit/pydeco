import numpy as np
from tqdm import tqdm

from pydeco.constants import *
from pydeco.problem.centralized_lq import CeLQ
from pydeco.problem.distributed_lq import DiLQ, create_distributed_envs
from pydeco.controller.lqr import LQR
from pydeco.controller.dlqr import DiLQR


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
    s0_0 = np.array([1, 0, 0, 0, 0])
    s0_1 = np.array([2, 0, 0, 0, 0])
    s0_2 = np.array([3, 0, 0, 0, 0])
    s0 = np.concatenate((s0_0, s0_1, s0_2))
    lqr = LQR()
    lqr.train(lq, method=TrainMethod.ANALYTICAL, initial_state=s0)
    P_star = lqr.P
    K_star = lqr.K
    # print(f'P: {P_star}')
    print(f'K: {K_star}')

    # solve DISTRIBUTED

    # create envs
    envs = create_distributed_envs(
        n_agents, communication_map, coupled_dynamics, coupled_rewards, A, B, Q, R)

    # initial state
    initial_states = [s0_0, s0_1, s0_2]

    curr_states = {
        i: env.reset(initial_state) for i, (env, initial_state) in enumerate(zip(envs, initial_states))
    }

    agents = [DiLQR() for _ in range(n_agents)]

    # initialize
    for env, agent in zip(envs, agents):
        agent.initialize_qlearn_ls(env.n_s, env.n_a)

    # training params
    max_policy_improves = 20
    max_policy_evals = 50
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

    print(agent.K)


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

    run_experiment()
