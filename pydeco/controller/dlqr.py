import logging
from time import perf_counter
from queue import Queue

import numpy as np

from pydeco.types import *
from pydeco.constants import PolicyType, TrainMethod, NoiseShape
from pydeco.problem.distributed_lq import DiLQ
from pydeco.controller.lqr import LQR


class DiLQR(LQR):
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
    ):
        n_q = n_s + n_a
        self._n_s = n_s
        self._n_q = n_q

        p = int(n_q * (n_q + 1) / 2)
        self._p = p

        self._noise_params = (np.zeros((n_a,)), np.eye(n_a), (1000,))

        self.K = np.full((n_a, n_s), fill_value=-.01)

        self._theta = np.full((p, 1), fill_value=.0)

    def reset_covar(self):
        beta = 1
        self._G_k = np.eye(self._p) * beta

    def eval_policy(
            self,
            env: DiLQ,
            agent_id: int,
            curr_states,
            gamma: Scalar,
    ):
        assert agent_id == env.agent_id

        # s
        curr_state = curr_states[agent_id]

        # z - receive info from neighboring agents
        curr_infos = []
        for j in env.neighbors:
            curr_infos.append(curr_states[j])

        # a
        curr_action = self.act(curr_state, information=curr_infos, policy_type=PolicyType.EPS_GREEDY)

        # r, (s', z')
        curr_reward, next_state = env.step(curr_action, information=curr_infos)

        # max_a'
        next_action = self.act(next_state, policy_type=PolicyType.GREEDY)

        # features from (s,a)
        f_x = self._build_feature_vector(curr_state, curr_action, self._p)
        f_x_next = self._build_feature_vector(next_state, next_action, self._p)

        # update params theta w/ RLS
        phi = f_x - gamma * f_x_next

        num = self._G_k @ phi * (curr_reward - phi.T @ self._theta)
        den = 1 + phi.T @ self._G_k @ phi
        theta_adj = num / den
        self._theta += theta_adj

        num2 = self._G_k @ phi @ phi.T @ self._G_k
        G_adj = num2 / den
        self._G_k -= G_adj

        # update state
        # TODO part of next state
        # curr_states[agent_id] = next_state[:env._sa_n_s]
        return next_state[:env._sa_n_s]

        # update counter
        # pi_eval_iter += 1

        # convergence
        # theta_adj_diff = np.max(np.abs(theta_adj))
        # pi_eval_converged = theta_adj_diff < eps

    def improve_policy(self, eps=1e-6) -> bool:
        # policy improvement
        H_k = self._convert_to_parameter_matrix(self._theta, self._n_q)

        n_s = self._n_s

        H_uk = H_k[n_s:, :n_s]
        H_uu = H_k[n_s:, n_s:]

        # argmax policy
        K_new = -np.linalg.inv(H_uu) @ H_uk

        # update counter
        # pi_improve_iter += 1

        # convergence
        # TODO consider stop at |P_new - P|
        converged = np.max(np.abs(K_new - self._K)) < eps

        self._K = K_new

        return converged
