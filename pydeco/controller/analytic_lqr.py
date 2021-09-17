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
    ):
        self._env = LQ(A, B, Q, R)

        # caching for logs an d viz
        self._cache = {}

    def solve(
            self,
            x0,
            t0,
            tn,
            n_steps,
    ) -> Tuple[Tensors, float, Dict]:
        # starting x0
        self._env.reset(x0)

        # clear previous results
        self._cache.clear()

        # solve DARE for Discrete-Time Finite-Horizon LQ
        time_grid = jnp.linspace(t0, tn, num=n_steps + 1)

        # calculate P, K
        A, B, Q, R = self._env.get_model()
        P = Q
        for k in time_grid[:-1]:
            pa = P @ A
            pb = P @ B
            P = Q + A.T @ pa - A.T @ pb @ inv(R + B.T @ pb) @ B.T @ pa

        K = - inv(R + B.T @ P @ B) @ B.T @ P @ A

        # calculate optimal controls
        us = []
        total_cost = .0
        x_k = x0
        for k in time_grid[:-1]:
            # optimal control at k
            u_k = K @ x_k
            us.append(u_k)

            # update state
            x_k, r_k = self._env.step(u_k)

            # increment stage cost
            total_cost += r_k

        rf = self._env.terminal_cost(x_k)
        total_cost += rf

        us = jnp.stack(us)

        return us, total_cost, self._cache
