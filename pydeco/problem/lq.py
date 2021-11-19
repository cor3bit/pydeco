import numpy as np

from pydeco.problem.env import Env
from pydeco.types import *


class LQ(Env):
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

        n_, m = B.shape
        assert n == n_
        self._n_u = m

        q_n, q_n_ = Q.shape
        assert q_n == n
        assert q_n_ == n

        r_m, r_m_ = R.shape
        assert r_m == m
        assert r_m_ == m

        # save params
        self._A = A
        self._B = B
        self._Q = Q
        self._R = R

        # initial guess
        self._x = None

    @property
    def n_x(self):
        return self._n_x

    @property
    def n_u(self):
        return self._n_u

    def reset(self, x0: Tensor):
        assert np.prod(x0.shape) == self._n_x
        self._x = x0.reshape((self._n_x, 1))
        return self._x

    def step(
            self,
            u: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # stage cost
        r_k = self._x.T @ self._Q @ self._x + u.T @ self._R @ u
        r_k = r_k.reshape((-1,))[0]

        # update state
        next_x_k = self._A @ self._x + self._B @ u
        self._x = next_x_k

        # return resulting state & cost
        return r_k, next_x_k

    def terminal_cost(
            self,
            xf: Tensor = None,
    ) -> Tensor:
        x = self._x if xf is None else xf
        return x.T @ self._Q @ x

    def get_model(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._A, self._B, self._Q, self._R

    def get_state(self) -> Tensor:
        return self._x

