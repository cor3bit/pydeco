import numpy as np

from pydeco.problem.lq import LQ
from pydeco.types import *


def create_distributed_envs(
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
    return [DiLQ(i, communication_map[i], coupled_dynamics, coupled_rewards,
                 system_matrix, control_matrix, state_reward_matrix, action_reward_matrix,
                 ) for i in range(n_agents)]


class DiLQ(LQ):
    def __init__(
            self,

            # communication topology
            agent_id: int,
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

        self._agent_id = agent_id
        self._neighbors = neighborhood_agents
        self._n_neighbors = len(neighborhood_agents)

        # sa_n_s, sa_n_a = control_matrix.shape
        # self._sa_n_s = sa_n_s
        # self._sa_n_a = sa_n_a

        # dynamics
        n_agents = 1 + self._n_neighbors
        I_n = np.eye(n_agents)
        S_n = np.zeros_like(I_n)
        S_n[0, 0] = 1

        if coupled_dynamics:
            raise NotImplementedError
        else:
            A = system_matrix
            B = control_matrix
            # A = np.kron(I_n, system_matrix)
            # B = np.kron(S_n, control_matrix)

        # rewards
        if coupled_rewards:
            Q = self._construct_coupled_Q(state_reward_matrix)
            R = action_reward_matrix
            # R = np.kron(S_n, action_reward_matrix)
        else:
            raise NotImplementedError
            # Q = state_reward_matrix
            # R = action_reward_matrix

        super().__init__(A, B, Q, R, check_dimensions=False)

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def neighbors(self):
        return self._neighbors

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
            information: Tensors = None,
            **kwargs
    ) -> Scalar:
        state_info = np.concatenate((self.get_state(), *information))
        r = state_info.T @ self._Q @ state_info + action.T @ self._R @ action
        return r.item()

    def _construct_coupled_Q(
            self,
            Q: Tensor,
    ):
        # Q_ construction
        n_agents = 1 + self._n_neighbors

        # ones on the diagonal (L + I_n)
        modified_laplacian = np.eye(n_agents)

        # subtract adjacency matrix
        modified_laplacian[:, 0] = -1.
        modified_laplacian[0, :] = -1.

        # main agent (degree + 1)
        modified_laplacian[0, 0] = n_agents

        Q_ = np.kron(modified_laplacian, Q)

        return Q_
