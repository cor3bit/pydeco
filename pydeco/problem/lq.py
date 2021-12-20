import numpy as np

from pydeco.problem.env import Env
from pydeco.types import *


class LQ(Env):
    def __init__(
            self,

            # LQ params
            system_matrix: Tensor,
            control_matrix: Tensor,
            state_reward_matrix: Tensor,
            action_reward_matrix: Tensor,

            # compatibility with CeLQ and DiLQ
            check_dimensions: bool = True,
    ):
        # save params
        self._A = system_matrix
        self._B = control_matrix
        self._Q = state_reward_matrix
        self._R = action_reward_matrix

        # dimensions
        n, m = self._B.shape
        self._n_s = n
        self._n_a = m

        # dimension check
        if check_dimensions:
            self._check_model_dimensions()
            self._check_reward_dimensions()

    def _check_model_dimensions(self):
        n, n_ = self._A.shape
        assert n == n_
        n_, m = self._B.shape
        assert n == n_

    def _check_reward_dimensions(self):
        q_n, q_n_ = self._Q.shape
        assert q_n == self._n_s
        assert q_n_ == self._n_s

        r_m, r_m_ = self._R.shape
        assert r_m == self._n_a
        assert r_m_ == self._n_a

    @property
    def n_s(self):
        return self._n_s

    @property
    def n_a(self):
        return self._n_a

    def reset(
            self,
            initial_state: Tensor,
            **kwargs
    ) -> Tensor:
        # assert np.prod(initial_state.shape) == self._n_s
        # self._state = initial_state.reshape((self._n_s, 1))
        self._state = initial_state.reshape((-1, 1))
        return self._state

    def terminal_cost(
            self,
            final_state: Tensor = None,
    ) -> Scalar:
        s = self._state if final_state is None else final_state
        r = s.T @ self._Q @ s
        return r.item()

    def get_model(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._A, self._B, self._Q, self._R

    def get_state(self) -> Tensor:
        return self._state

    def _transition_fn(
            self,
            action: Tensor,
            **kwargs
    ) -> Tensor:
        return self._A @ self._state + self._B @ action

    def _reward_fn(
            self,
            action: Tensor,
            **kwargs
    ) -> Scalar:
        r = self._state.T @ self._Q @ self._state + action.T @ self._R @ action
        return r.item()
