import logging
from time import perf_counter
from queue import Queue

import numpy as np
from tqdm import tqdm

from pydeco.types import *
from pydeco.constants import *
from pydeco.problem.distributed_lq import LocalLQ, MultiAgentLQ
from pydeco.controller.lqr import LQR
from pydeco.controller.agent import MultiAgent


class MultiAgentLQR(MultiAgent):
    def __init__(
            self,
            n_agents: int,
            verbose: bool = True,
    ):
        self._agent_map = {i: LocalLQR(verbose=verbose) for i in range(n_agents)}

        # logging
        self._configure_logger('D-LQR', verbose)

        # caching
        self._cache = {}

        # training related
        self._calibrated = False

    def train(
            self,
            ma_env: MultiAgentLQ,
            method: str,
            policy_eval: str = PolicyEvaluation.QLEARN_RLS,

            gamma: Scalar = 1.,
            eps: Scalar = 1e-8,
            alpha: Scalar = 1e-3,

            max_policy_evals: int = 100,
            max_policy_improves: int = 10,
            reset_every_n: int = 100,

            initial_states: Tensors | None = None,
            initial_state_fn: Callable = None,
            initial_policies: Tensors | None = None,
            sa_initial_policy: Tensor | None = None,

            optimal_controller: Tensor | None = None,

            **kwargs
    ):
        if self._calibrated:
            self._logger.warning('MA controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating MA controller.')

        # optimal policy for logging
        self._K_star = optimal_controller

        # find optimal policy
        match method:
            case TrainMethod.DARE:  # TODO check that a viable solution
                self._train_analytical_dare(
                    ma_env,
                    gamma=gamma,
                    **kwargs
                )
            case TrainMethod.GPI:
                self._train_gpi(
                    ma_env=ma_env,
                    policy_eval=policy_eval,
                    gamma=gamma,
                    eps=eps,
                    alpha=alpha,
                    max_policy_evals=max_policy_evals,
                    max_policy_improves=max_policy_improves,
                    reset_every_n=reset_every_n,
                    initial_states=initial_states,
                    initial_state_fn=initial_state_fn,
                    sa_initial_policy=sa_initial_policy,
                    initial_policies=initial_policies,
                    **kwargs
                )
            case _:
                raise ValueError(f'Method {method} not supported.')

    def simulate_trajectory(
            self,
            ma_env: MultiAgentLQ,
            initial_states: Tensors,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float]:
        # self._logger.info(f'Simulating a trajectory.')
        ma_env.reset(initial_states)

        time_grid = np.linspace(t0, tn, num=n_steps + 1)

        xs = []
        us = []
        total_cost = .0

        for k in time_grid[:-1]:

            step_xs = []
            step_us = []

            for local_env, (agent_id, agent) in zip(ma_env._env_map.values(), self._agent_map.items()):
                curr_state = ma_env.get_state(agent_id)
                curr_info = ma_env.get_info(agent_id)
                curr_action = agent.act(curr_state, information=curr_info, policy_type=PolicyType.GREEDY)

                step_xs.append(curr_state)
                step_us.append(curr_action)

                # r, s'
                curr_reward, next_state = local_env.step(curr_action, information=curr_info)
                ma_env.set_next_state(agent_id, next_state)

                total_cost += curr_reward

            ma_env.roll_states()

            xs.append(np.concatenate(step_xs))
            us.append(np.concatenate(step_us))

        # last step
        step_xs = []
        for agent_id, agent in self._agent_map.items():
            curr_state = ma_env.get_state(agent_id)
            step_xs.append(curr_state)

        xs.append(np.concatenate(step_xs))

        # TODO include terminal cost

        xs = np.squeeze(np.stack(xs))
        us = np.squeeze(np.stack(us))

        return xs, us, total_cost

    def _train_analytical_dare(
            self,
            ma_env: MultiAgentLQ,
            gamma: Scalar,
            **kwargs
    ):
        for env, agent in zip(ma_env._env_map.values(), self._agent_map.values()):
            agent.train(env, TrainMethod.DARE, gamma=gamma)

            # save only a strip of K related to the main agent
            agent.K = agent.K[:env.n_a, :]

    def _train_gpi(
            self,
            ma_env: MultiAgentLQ,
            policy_eval: str,
            gamma: Scalar,
            eps: Scalar,
            alpha: Scalar,
            max_policy_evals: int,
            max_policy_improves: int,
            reset_every_n: int,
            initial_states: Tensors | None,
            initial_state_fn: Callable,
            initial_policies: Tensors | None,
            sa_initial_policy: Tensor | None,
            **kwargs
    ):
        # initialize
        ma_env.reset(initial_states, generating_fn=initial_state_fn)

        for env, agent in zip(ma_env._env_map.values(), self._agent_map.values()):
            agent.initialize_qlearn_ls(env.n_s, env.n_a, env.n_neighbors, sa_initial_policy)

        # policy improvement loop
        all_converged = False
        iter = 1

        self._log_header()

        # pbar = tqdm(total=max_policy_improves * max_policy_evals)
        while (not all_converged) and (not iter > max_policy_improves):
            # reset covar at the beginning of a new policy eval loop
            for agent in self._agent_map.values():
                agent.reset_covar()

            # POLICY EVAL
            for i_eval in range(max_policy_evals):
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
                    # s, z, a, s', z'
                    curr_state = ma_env.get_state(agent_id)
                    curr_info = ma_env.get_info(agent_id)
                    curr_action = ma_env.get_action(agent_id)
                    curr_reward = ma_env.get_reward(agent_id)
                    next_state = ma_env.get_next_state(agent_id)
                    next_info = ma_env.get_next_info(agent_id)

                    #  max_a'
                    next_action = agent.act(next_state, information=next_info, policy_type=PolicyType.GREEDY)

                    # construct features from (s, z, a)
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

                # reset problem to avoid state blow-up due to instable K
                if i_eval % reset_every_n == 0:
                    ma_env.reset(initial_states, generating_fn=initial_state_fn)
                # roll states (curr states <- next states)
                else:
                    ma_env.roll_states()

                # update progress
                # pbar.update(1)

            # POLICY IMPROVE
            all_converged = True
            for agent_id, agent in self._agent_map.items():
                converged, diff = agent.improve_policy(eps)
                all_converged &= converged

                self._save_param(agent_id, iter, 'max_diff', diff)
                self._log_step(agent_id, iter)

            # logging against optimal sln
            if self._K_star is not None:
                K_distr = self._reconstruct_full_K(ma_env)
                self._save_param('-', iter, 'opt_diff', np.linalg.norm((K_distr, self._K_star)))
                self._log_step('-', iter)

            # update iteration
            iter += 1

        # pbar.close()

    def _configure_logger(
            self,
            name: str,
            verbose: bool,
    ):
        self._logger = logging.getLogger(name)

        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing {name} controller.')

    def _save_param(
            self,
            agent_id: int | str,
            iter: int,
            name: str,
            value: Union[float, int, str, Tensor],
    ):
        key = (agent_id, iter)

        if key not in self._cache:
            self._cache[key] = {}

        if name in self._cache[key]:
            self._logger.error(f'{name} has already been cached for time step {key}!')

        self._cache[key][name] = value

    def _log_header(self):
        ai = 'agent'
        i = 'iter'
        max_diff = 'K_max_diff'
        opt_policy = '||K*-K||'

        msg = f"|{i:^7}|{ai:^7}|{max_diff:^12}|{opt_policy:^12}|"

        self._logger.info(msg)

    def _log_step(
            self,
            agent_id: int | str,
            iter: int,
    ):
        key = (agent_id, iter)
        agent_cache_at_iter = self._cache[key]

        max_diff = agent_cache_at_iter['max_diff'] if 'max_diff' in agent_cache_at_iter else np.NaN
        opt_policy = agent_cache_at_iter['opt_diff'] if 'opt_diff' in agent_cache_at_iter else np.NaN

        msg1 = f'|{iter:<7}|{agent_id:<7}|{max_diff:<12.5f}|{opt_policy:<12.5f}|'

        self._logger.info(msg1)

    def _reconstruct_full_K(self, ma_env: MultiAgentLQ):
        agent_policies = []

        n_agents = len(self._agent_map)

        for local_env, (agent_id, agent) in zip(ma_env._env_map.values(), self._agent_map.items()):
            n_s, n_a = local_env.n_s, local_env.n_a
            K_agent = agent.K

            K_strip = np.zeros((n_a, n_agents * n_s))

            K_strip[:, agent_id:agent_id + n_s] = K_agent[:, :n_s]

            for i, neighbor in enumerate(local_env.neighbors, start=1):
                K_strip[:, neighbor:neighbor + n_s] = K_agent[:, i:i + n_s]

            agent_policies.append(K_strip)

        return np.concatenate(agent_policies)


class LocalLQR(LQR):
    def __init__(
            self,
            noise_type: str = NoiseShape.MV_NORMAL,
            verbose: bool = True,
    ):
        super().__init__('LocalLQR', noise_type, verbose=verbose)

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

    def _init_gpi(
            self,
            env: LocalLQ,
            initial_state: Tensor,
            initial_policy: Tensor,
            noise_cache_size: int = 1000,
    ):
        # TODO!!

        # dimensions of the controller
        n_s = env.n_s
        n_a = env.n_a
        n_q = n_s + n_a
        p = int(n_q * (n_q + 1) / 2)

        self._n_s = n_s
        self._n_q = n_q
        self._p = p

        # noise for acting with exploration
        self._noise_params = (np.zeros((n_a,)), np.eye(n_a), (noise_cache_size,))

        # s_0
        env.reset(initial_state)

        # K_0
        self.K = initial_policy

        # weights of the FA
        self._weights = np.zeros((p, 1))


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

        # K_0
        self.K = np.tile(sa_initial_policy, n_agents)

        # init FA weights and policy
        self._weights = np.zeros((p, 1))

    # def reset_covar(self, beta: Scalar = 1.):
    #     self._G_k = np.eye(self._p) * beta
    #
    # def improve_policy(self, eps: Scalar) -> bool:
    #     # policy improvement
    #     H_k = self._convert_to_parameter_matrix(self._theta, self._n_q)
    #
    #     n_s = self._n_s
    #
    #     H_uk = H_k[n_s:, :n_s]
    #     H_uu = H_k[n_s:, n_s:]
    #
    #     # argmax policy
    #     K_new = -np.linalg.inv(H_uu) @ H_uk
    #
    #     # convergence
    #     # TODO consider stop at |P_new - P|
    #     diff = np.max(np.abs(K_new - self._K))
    #
    #     # print(diff)
    #
    #     converged = diff < eps
    #
    #     self._K = K_new
    #
    #     return converged, diff
