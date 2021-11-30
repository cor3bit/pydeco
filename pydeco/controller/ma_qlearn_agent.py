import logging
from time import perf_counter

import numpy as np

from pydeco.constants import PolicyType
from pydeco.types import *
from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ


class MaQlearnAgent(LQR):
    def __init__(
            self,
            verbose: bool = True,
    ):
        super().__init__('analytical_lqr', verbose)

    def act(
            self,
            x: Tensor,
            policy_type: PolicyType = PolicyType.GREEDY,
    ) -> Tensor:
        if policy_type != PolicyType.GREEDY:
            raise ValueError(f'Policy {policy_type} not supported.')

        return self._K @ x