import logging
from time import perf_counter

from jax.numpy.linalg import inv
import jax.numpy as jnp

from pydeco.types import *
from pydeco.controller.controller import Controller
from pydeco.problem.lq import LQ


class AnalyticalLQR(Controller):
    def __init__(
            self,
            A: Tensor,
            B: Tensor,
            Q: Tensor,
            R: Tensor,
            verbose: bool = True,
    ):
        # logger
        self._logger = logging.getLogger('analytical_lqr')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing controller.')
        self._cache = {}

        # env
        self._env = LQ(A, B, Q, R)

        # model
        self._P = None
        self._K = None

    def fit(
            self,
            n_steps: int = 10,
    ):
        if self._calibrated:
            self._logger.warning('Controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating controller.')

        # calculate P, K by solving DARE for Discrete-Time Finite-Horizon LQ
        A, B, Q, R = self._env.get_model()
        P = Q
        for _ in range(n_steps):
            pa = P @ A
            pb = P @ B
            P = Q + A.T @ pa - A.T @ pb @ inv(R + B.T @ pb) @ B.T @ pa

        K = - inv(R + B.T @ P @ B) @ B.T @ P @ A

        # save values
        self._P = P
        self._K = K
        self._calibrated = True

    def simulate_trajectory(
            self,
            x0: Tensor,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float, Dict]:
        if not self._calibrated:
            self._logger.warning('Controller has not been calibrated!')
            return tuple()

        self._logger.info(f'Simulating a trajectory.')

        # starting x0
        self._env.reset(x0)

        # clear previous results
        self._cache.clear()

        # solve DARE for Discrete-Time Finite-Horizon LQ
        time_grid = jnp.linspace(t0, tn, num=n_steps + 1)

        # calculate optimal controls
        xs = [x0]
        us = []
        total_cost = .0
        x_k = x0
        for k in time_grid[:-1]:
            # optimal control at k
            u_k = self._K @ x_k
            us.append(u_k)

            # update state
            x_k, r_k = self._env.step(u_k)
            xs.append(x_k)

            # increment stage cost
            total_cost += r_k

        rf = self._env.terminal_cost(x_k)
        total_cost += rf

        xs = jnp.stack(xs)
        us = jnp.stack(us)

        return xs, us, total_cost, self._cache
