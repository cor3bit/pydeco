from abc import ABC, abstractmethod
import logging

import numpy as np

from pydeco.types import *
from pydeco.problem.lq import LQ


class LQR(ABC):
    _cache = None
    _calibrated = False

    def __init__(
            self,
            controller_id: str,
            verbose: bool = True,
    ):
        # logger
        self._logger = logging.getLogger(f'{controller_id}')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing controller.')
        self._cache = {}

        # model
        # Value function matrix (n_x, n_x)
        self._P = None
        # Policy matrix (n_u, n_x)
        self._K = None

    def act(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def act_q(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def train(self, **kwargs):
        # find optimal policy
        raise NotImplementedError

    def simulate_trajectory(
            self,
            env: LQ,
            x0: Tensor,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float]:
        if not self._calibrated:
            self._logger.warning('Controller has not been calibrated!')
            return tuple()

        self._logger.info(f'Simulating a trajectory.')

        # starting x0
        env.reset(x0)

        # clear previous results
        self._cache.clear()

        # calculate optimal controls
        time_grid = np.linspace(t0, tn, num=n_steps + 1)
        x_k = env.get_state()
        xs = [x_k]
        us = []
        total_cost = .0
        for k in time_grid[:-1]:
            # optimal control at k
            u_k = self.act(x_k)
            us.append(u_k)

            # update state
            r_k, next_x_k = env.step(u_k)
            xs.append(x_k)

            # increment stage cost
            total_cost += r_k

            # update state
            x_k = next_x_k

        rf = env.terminal_cost(x_k)
        total_cost += rf

        xs = np.stack(xs)
        us = np.stack(us)

        return xs, us, total_cost
