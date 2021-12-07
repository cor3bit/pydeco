import numpy as np

from pydeco.problem.env import DistributedEnv
from pydeco.types import *


class DiLQ(DistributedEnv):
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
        # dimension check
        n, n_ = A.shape
        assert n == n_
        self._n_x = n

    def step(self, agent_id: int, action: Tensor, information: Tensors) -> Tuple[Tensor, Tensor]:
        pass





