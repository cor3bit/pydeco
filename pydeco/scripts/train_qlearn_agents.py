import time

import numpy as np
from tqdm import tqdm

from pydeco.problem.platoon import VehiclePlatoon
from pydeco.controller.ma_qlearn_agent import CollaboratingQlearnAgent


def get_info(state_n, neighbors):
    pass


def run_training():
    # params
    n_vehicles = 5

    # initial states
    leader_state = [70., 20., 0.]
    initial_states = [leader_state]
    for pos in [50, 30, 10, 0]:
        initial_states.append([pos, 0, 0])

    env = VehiclePlatoon(n_vehicles)

    agents = [CollaboratingQlearnAgent(i) for i in range(n_vehicles)]

    neighborhood = {
        0: None,
        1: [0, ],
        2: [1, ],
        3: [2, ],
        4: [3, ],
    }

    state_n = env.reset(initial_states)

    while True:
        pass

        for agent in agents:
            agent_id = agent.agent_id
            curr_state_i = state_n[agent_id]

            neighbors = neighborhood[agent_id]
            curr_info_i = get_info(state_n, neighbors)

            curr_action_i = agent.act(curr_state_i, curr_info_i)

            curr_reward_i, next_state_i = env.step(agent_id, curr_action_i)

    #
    # for i_episode in tqdm(range(N_EPISODES)):
    #     # init env
    #     # obs_n = env.reset()
    #     obs_n = env.reset_default()
    #     env.render()
    #
    #     # init agents
    #     agents = create_agents(env, AGENT_TYPE)
    #
    #     # init stopping condition
    #     done_n = [False] * env.n_agents
    #
    #     total_reward = 0.
    #
    #     # run an episode until all prey is caught
    #     while not all(done_n):
    #         prev_actions = {}
    #         act_n = []
    #         for i, (agent, obs) in enumerate(zip(agents, obs_n)):
    #             action_id = agent.act(obs, prev_actions=prev_actions)
    #
    #             prev_actions[i] = action_id
    #             act_n.append(action_id)
    #
    #         # update step
    #         obs_n, reward_n, done_n, info = env.step(act_n)
    #
    #         total_reward += np.sum(reward_n)
    #
    #         time.sleep(0.5)
    #         env.render()
    #
    #     print(f'Episode {i_episode}: Avg Reward is {total_reward / env.n_agents}')
    #
    # time.sleep(2.)

    env.close()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    np.random.seed(42)

    run_training()
