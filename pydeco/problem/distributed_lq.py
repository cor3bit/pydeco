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

        # dynamics
        if coupled_dynamics:
            raise NotImplementedError
        else:
            A, B = system_matrix, control_matrix

        # rewards
        if coupled_rewards:
            Q = self._construct_coupled_Q(state_reward_matrix)
            R = action_reward_matrix
        else:
            Q = state_reward_matrix
            R = action_reward_matrix

        super().__init__(A, B, Q, R, check_dimensions=False)

    @property
    def agent_id(self):
        return self._agent_id

    def step(
            self,
            action: Tensor,
            information: Tensors = None,
            **kwargs
    ) -> Tuple[Scalar, Tensor]:
        return super().step(action, information=information)

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
