import numpy as np

from pydeco.problem.env import DistributedEnv
from pydeco.types import *


class DiLQ(DistributedEnv):
    def __init__(
            self,
            A: Tensor,
            B: Tensor,
            Q: Tensor,
            R: Tensor,
    ):
        # dimension check
        n, n_ = A.shape
        assert n == n_
        self._n_x = n

    def step(self, agent_id: int, action: Tensor, information: Tensors) -> Tuple[Tensor, Tensor]:
        pass





