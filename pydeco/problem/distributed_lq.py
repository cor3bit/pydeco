import numpy as np

from pydeco.problem.lq import LQ
from pydeco.problem.env import MultiAgentEnv
from pydeco.types import *


class MultiAgentLQ(MultiAgentEnv):
    def __init__(
            self,
            n_agents: int,
            communication_map: dict[int, Sequence[int]],
            coupled_dynamics: bool,
            coupled_rewards: bool,

            # individual LQ params
            system_matrix: Tensor,
            control_matrix: Tensor,
            state_reward_matrix: Tensor,
            action_reward_matrix: Tensor,
    ):
        # create envs
        self._env_map = {
            i: LocalLQ(
                communication_map[i], coupled_dynamics, coupled_rewards,
                system_matrix, control_matrix, state_reward_matrix, action_reward_matrix,
            ) for i in range(n_agents)
        }

        # local cache
        self._n_agents = n_agents
        self._curr_states = None
        self._next_states = None
        self._curr_rewards = None
        self._curr_actions = None

    def step(
            self,
            agent_id: int,
            action: Tensor,
            information: Tensor | None = None,
            **kwargs
    ) -> Sequence[Tuple[Scalar, Tensor]]:
        raise NotImplementedError

    def reset(
            self,
            initial_states: Tensors,
            generating_fn: Callable = None,
            **kwargs
    ):
        if generating_fn is not None:
            n_agents = len(self._env_map)
            initial_states = [generating_fn() for _ in range(n_agents)]

        # individual env
        states = [
            env.reset(initial_state) for env, initial_state in zip(
                self._env_map.values(), initial_states)
        ]

        # MA cache as a message broker
        self._curr_states = {i: initial_state for i, initial_state in enumerate(states)}
        self._next_states = {i: None for i in range(self._n_agents)}
        self._curr_rewards = {i: None for i in range(self._n_agents)}
        self._curr_actions = {i: None for i in range(self._n_agents)}

    def roll_states(self):
        self._curr_states = self._next_states
        self._next_states = {i: None for i in range(self._n_agents)}

    def get_state(
            self,
            agent_id: int,
    ) -> Tensor:
        return self._curr_states[agent_id]

    def get_next_state(
            self,
            agent_id: int,
    ) -> Tensor:
        return self._next_states[agent_id]

    def get_action(
            self,
            agent_id: int,
    ) -> Tensor:
        return self._curr_actions[agent_id]

    def get_info(
            self,
            agent_id: int,
    ) -> Tensors:
        neighbors = self._env_map[agent_id].neighbors
        return [self.get_state(neighbor_id) for neighbor_id in neighbors]

    def get_next_info(
            self,
            agent_id: int,
    ) -> Tensors:
        neighbors = self._env_map[agent_id].neighbors
        return [self.get_next_state(neighbor_id) for neighbor_id in neighbors]

    def get_reward(
            self,
            agent_id: int,
    ) -> Scalar:
        return self._curr_rewards[agent_id]

    def set_reward(
            self,
            agent_id: int,
            reward: Scalar,
    ):
        self._curr_rewards[agent_id] = reward

    def set_action(
            self,
            agent_id: int,
            action: Tensor,
    ):
        self._curr_actions[agent_id] = action

    def set_next_state(
            self,
            agent_id: int,
            next_state: Tensor,
    ):
        self._next_states[agent_id] = next_state


class LocalLQ(LQ):
    def __init__(
            self,

            # communication topology
            # agent_id: int,
            neighborhood_agents: Sequence[int],
            coupled_dynamics: bool,
            coupled_rewards: bool,

            # individual LQ params
            system_matrix: Tensor,
            control_matrix: Tensor,
            state_reward_matrix: Tensor,
            action_reward_matrix: Tensor,
    ):
        assert not coupled_dynamics
        assert coupled_rewards

        # self._agent_id = agent_id
        # self._neighbors = neighborhood_agents
        # self._n_neighbors = len(neighborhood_agents)

        # sa_n_s, sa_n_a = control_matrix.shape
        # self._sa_n_s = sa_n_s
        # self._sa_n_a = sa_n_a

        # dynamics
        n_neighbors = len(neighborhood_agents)
        n_agents = 1 + n_neighbors

        self._neighbors = neighborhood_agents
        self._n_neighbors = n_neighbors

        # I_n = np.eye(n_agents)
        # S_n = np.zeros_like(I_n)
        # S_n[0, 0] = 1

        if coupled_dynamics:
            raise NotImplementedError
        else:
            A = system_matrix
            B = control_matrix
            # A = np.kron(I_n, system_matrix)
            # B = np.kron(S_n, control_matrix)

        # rewards
        if coupled_rewards:
            Q = self._construct_coupled_Q(state_reward_matrix, n_agents)
            R = action_reward_matrix
            # R = np.kron(S_n, action_reward_matrix)
        else:
            raise NotImplementedError
            # Q = state_reward_matrix
            # R = action_reward_matrix

        super().__init__(A, B, Q, R, check_dimensions=False)

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def n_neighbors(self):
        return self._n_neighbors

    # @property
    # def agent_id(self):
    #     return self._agent_id
    #
    # @property
    # def neighbors(self):
    #     return self._neighbors

    # def step(
    #         self,
    #         action: Tensor,
    #         information: Tensors = None,
    #         **kwargs
    # ) -> Tuple[Scalar, Tensor]:
    #     # get current reward
    #     curr_reward = self._reward_fn(action, information=information, **kwargs)
    #
    #     # transition to next state
    #     next_state = self._transition_fn(action, information=information, **kwargs)
    #
    #     # update internal state
    #     self._state = next_state[:self._sa_n_s]
    #
    #     # return resulting state & reward
    #     return curr_reward, next_state

    # def _transition_fn(
    #         self,
    #         action: Tensor,
    #         information: Tensors = None,
    #         **kwargs
    # ) -> Tensor:
    #     state_info = np.concatenate((self.get_state(), *information))
    #     return self._A @ state_info + self._B @ action

    def _reward_fn(
            self,
            action: Tensor,
            information: Tensors | None = None,
            **kwargs
    ) -> Scalar:
        state_info = np.concatenate((self.get_state(), *information))
        r = state_info.T @ self._Q @ state_info + action.T @ self._R @ action
        return r.item()

    def _construct_coupled_Q(
            self,
            Q: Tensor,
            n_agents: int,
    ):
        # Q_ construction
        # n_agents = 1 + self._n_neighbors

        # ones on the diagonal (L + I_n)
        modified_laplacian = np.eye(n_agents)

        # subtract adjacency matrix
        modified_laplacian[:, 0] = -1.
        modified_laplacian[0, :] = -1.

        # main agent (degree + 1)
        modified_laplacian[0, 0] = n_agents

        Q_ = np.kron(modified_laplacian, Q)

        return Q_
