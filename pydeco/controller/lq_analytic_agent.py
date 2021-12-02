import logging
from time import perf_counter

import numpy as np

from pydeco.constants import PolicyType
from pydeco.types import *
from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ
from pydeco.problem.centralized_lq import CeLQ


class AnalyticalLQR(LQR):
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

    def act_q(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def train(
            self,
            env: LQ | CeLQ,
            n_steps: int = 10,
    ):
        if self._calibrated:
            self._logger.warning('Controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating controller.')

        # analytical solution requires access to the LQ model
        A, B, Q, R = env.get_model()

        # calculate P, K by solving DARE for Discrete-Time Finite-Horizon LQ
        P = Q
        for _ in range(n_steps):
            pa = P @ A
            pb = P @ B
            P = Q + A.T @ pa - A.T @ pb @ np.linalg.inv(R + B.T @ pb) @ B.T @ pa

        K = - np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        # save calibrated values
        self._P = P
        self._K = K
        self._calibrated = True
