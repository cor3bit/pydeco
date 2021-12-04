import logging

import numpy as np

from pydeco.types import *
from pydeco.problem.distributed_lq import DiLQ
from pydeco.controller.agent import CollaboratingAgent


class DLQR(CollaboratingAgent):
    def __init__(
            self,
            agent_id: int,
            verbose: bool = True,
    ):
        # super().__init__('ma_qlearn_agent', verbose)
        self._agent_id = agent_id

    @property
    def agent_id(self):
        return self._agent_id

    # def act(
    #         self,
    #         x: Tensor,
    #         policy_type: PolicyType = PolicyType.GREEDY,
    # ) -> Tensor:
    #     if policy_type != PolicyType.GREEDY:
    #         raise ValueError(f'Policy {policy_type} not supported.')
    #
    #     return self._K @ x
    #
    # def train(self):
    #     pass
