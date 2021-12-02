import numpy as np

from pydeco.problem.lq import LQ
from pydeco.types import *


class CeLQ(LQ):
    # ASSUMPTION: decoupled dynamics, coupled rewards (eg. borrelli08, alemzadeh18)

    def __init__(
            self,

            # communication topology
            n_agents: int,
            communication_links: Sequence[Tuple],

            # individual LQ params
            system_matrix: Tensor,
            control_matrix: Tensor,
            state_reward_matrix: Tensor,
            action_reward_matrix: Tensor,
    ):
        A, B, Q, R = self._convert_to_ma_model(
            n_agents, communication_links, system_matrix,
            control_matrix, state_reward_matrix, action_reward_matrix,
        )

        super().__init__(A, B, Q, R)

    def _convert_to_ma_model(
            self,
            n_agents: int,
            edges: Sequence[Tuple],
            A: Tensor,
            B: Tensor,
            Q: Tensor,
            R: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        I_n = np.eye(n_agents)
        A_ = np.kron(I_n, A)
        B_ = np.kron(I_n, B)
        R_ = np.kron(I_n, R)

        # L - Laplacian, L(G) = D(G) - A(G)
        A_G = np.zeros((n_agents, n_agents))
        for i, j in edges:
            A_G[i - 1, j - 1] = 1.

        D_G = np.diag(np.sum(A_G, axis=1))

        L_G = D_G - A_G

        Q_ = np.kron(L_G + I_n, Q)

        return A_, B_, Q_, R_
