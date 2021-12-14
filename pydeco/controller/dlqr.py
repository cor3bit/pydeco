import logging
from time import perf_counter
from queue import Queue

import numpy as np
from tqdm import tqdm

from pydeco.types import *
from pydeco.constants import PolicyType, TrainMethod, NoiseShape
from pydeco.problem.distributed_lq import LocalLQ, MultiAgentLQ
from pydeco.controller.lqr import LQR
from pydeco.controller.agent import MultiAgent


class MultiAgentLQR(MultiAgent):
    def __init__(
            self,
            n_agents: int,
    ):
        self._agent_map = {i: LocalLQR() for i in range(n_agents)}

    def train(
            self,
            ma_env: MultiAgentLQ,
            gamma: Scalar = 1.,
            eps: Scalar = 1e-7,
            max_policy_evals: int = 200,
            max_policy_improves: int = 10,
            initial_states: Tensors | None = None,
            sa_initial_policy: Tensor | None = None,
            initial_policies: Tensors | None = None,
            alpha: Scalar = 0.01,
    ):
        # initialize
        ma_env.reset(initial_states)

        for env, agent in zip(ma_env._env_map.values(), self._agent_map.values()):
            agent.initialize_qlearn_ls(env.n_s, env.n_a, env.n_neighbors, sa_initial_policy)

        # policy improvement loop
        all_converged = False
        iter = 0

        pbar = tqdm(total=max_policy_improves * max_policy_evals)
        while not all_converged and iter < max_policy_improves:
            # TODO REMOVE initialize
            ma_env.reset(initial_states)

            # reset covar at the beginning of a new policy eval loop
            for agent in self._agent_map.values():
                agent.reset_covar()

            # policy evaluation loop
            for _ in range(max_policy_evals):
                # make a step with all agents based on the local information
                for local_env, (agent_id, agent) in zip(ma_env._env_map.values(), self._agent_map.items()):
                    # s, z, a
                    curr_state = ma_env.get_state(agent_id)
                    curr_info = ma_env.get_info(agent_id)
                    curr_action = agent.act(curr_state, information=curr_info, policy_type=PolicyType.EPS_GREEDY)

                    # r, s'
                    curr_reward, next_state = local_env.step(curr_action, information=curr_info)

                    # cache rewards & next states
                    ma_env.set_reward(agent_id, curr_reward)
                    ma_env.set_action(agent_id, curr_action)
                    ma_env.set_next_state(agent_id, next_state)

                # update weights with RLS
                for local_env, (agent_id, agent) in zip(ma_env._env_map.values(), self._agent_map.items()):
                    # s, z, a
                    curr_state = ma_env.get_state(agent_id)
                    curr_info = ma_env.get_info(agent_id)
                    curr_action = ma_env.get_action(agent_id)
                    curr_reward = ma_env.get_reward(agent_id)

                    # s', z', max_a'
                    next_state = ma_env.get_next_state(agent_id)
                    next_info = ma_env.get_next_info(agent_id)
                    next_action = agent.act(next_state, information=next_info, policy_type=PolicyType.GREEDY)

                    # construct features from (s, z, a)
                    # TODO build feature with *args
                    curr_features = agent._build_feature_vector(agent._p, curr_state, *curr_info, curr_action)
                    next_features = agent._build_feature_vector(agent._p, next_state, *next_info, next_action)
                    phi = curr_features - gamma * next_features

                    # update theta
                    num = agent._G_k @ phi * (curr_reward - phi.T @ agent._theta)
                    den = 1 + phi.T @ agent._G_k @ phi
                    theta_adj = num / den
                    agent._theta += theta_adj

                    # update covar matrix
                    num2 = agent._G_k @ phi @ phi.T @ agent._G_k
                    G_adj = num2 / den
                    agent._G_k -= G_adj

                # roll states (curr states <- next states)
                ma_env.roll_states()

                # update progress
                pbar.update(1)

            # policy improvement step
            all_converged = True
            # TODO check improve dimensions
            for agent in self._agent_map.values():
                converged = agent.improve_policy(eps)
                all_converged &= converged

            # update iteration
            iter += 1

        pbar.close()

    def simulate_trajectory(
            self,
            env: MultiAgentLQ,
            initial_states: Tensors,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float]:
        pass


class LocalLQR(LQR):
    def __init__(
            self,
            noise_type: str = NoiseShape.MV_NORMAL,
            verbose: bool = True,
    ):
        super().__init__('LocalLQR', noise_type, verbose)

    def act(
            self,
            state: Tensor,
            information: Tensors = None,
            policy_type: PolicyType = PolicyType.GREEDY,
            **kwargs
    ) -> Tensor:
        if information is not None:
            state_with_info = np.concatenate([state, *information])
        else:
            state_with_info = state

        return super().act(state_with_info, policy_type, **kwargs)

    def initialize_qlearn_ls(
            self,
            n_s: int,
            n_a: int,
            n_neighbors: int,
            sa_initial_policy: Tensor,
    ):
        n_agents = 1 + n_neighbors
        n_q = n_s * n_agents + n_a

        self._n_s = n_s * n_agents
        self._n_q = n_q
        p = int(n_q * (n_q + 1) / 2)
        self._p = p

        # noise for act() function
        self._noise_params = (np.zeros((n_a,)), np.eye(n_a), (1000,))

        # init FA weights and policy
        self._theta = np.full((p, 1), fill_value=.0)
        self.K = np.tile(sa_initial_policy, n_agents)

    def reset_covar(self, beta: Scalar = 1.):
        self._G_k = np.eye(self._p) * beta

    def improve_policy(self, eps: Scalar) -> bool:
        # policy improvement
        H_k = self._convert_to_parameter_matrix(self._theta, self._n_q)

        n_s = self._n_s

        H_uk = H_k[n_s:, :n_s]
        H_uu = H_k[n_s:, n_s:]

        # argmax policy
        K_new = -np.linalg.inv(H_uu) @ H_uk

        # convergence
        # TODO consider stop at |P_new - P|
        diff = np.max(np.abs(K_new - self._K))

        print(diff)

        converged = diff < eps

        self._K = K_new

        return converged
