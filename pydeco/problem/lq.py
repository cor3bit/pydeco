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
        self._A = A
        self._B = B
        self._Q = Q
        self._R = R

        # TODO dimension check
        self._x = None

    def reset(self, x0: Tensor):
        # TODO dimension check
        # TODO set x (state), reshape as (n_x, 1)

        self._x = x0

    def step(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        # update state
        new_x = self._model(self._x, u)
        self._x = new_x

        # stage cost
        stage_cost = self._x.T @ self._Q @ self._x + u.T @ self._R @ u

        # return resulting state & cost
        return new_x, stage_cost

    def terminal_cost(self, xf: Tensor = None) -> Tensor:
        x = self._x if xf is None else xf
        return x.T @ self._Q @ x

    def get_model(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._A, self._B, self._Q, self._R

    def _model(self, x: Tensor, u: Tensor, t: Optional[float] = None) -> Tensor:
        return self._A @ x + self._B @ u
