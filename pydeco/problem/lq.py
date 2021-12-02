import numpy as np

from pydeco.problem.env import Env
from pydeco.types import *


class LQ(Env):
    def __init__(
            self,
            system_matrix: Tensor,
            control_matrix: Tensor,
            state_reward_matrix: Tensor,
            action_reward_matrix: Tensor,
    ):
        # save params
        self._A = system_matrix
        self._B = control_matrix
        self._Q = state_reward_matrix
        self._R = action_reward_matrix

        # dimension check
        self._check_dimensions()

    def _check_dimensions(self):
        n, n_ = self._A.shape
        assert n == n_
        self._n_s = n

        n_, m = self._B.shape
        assert n == n_
        self._n_a = m

        q_n, q_n_ = self._Q.shape
        assert q_n == n
        assert q_n_ == n

        r_m, r_m_ = self._R.shape
        assert r_m == m
        assert r_m_ == m

    @property
    def n_s(self):
        return self._n_s

    @property
    def n_a(self):
        return self._n_a

    def reset(
            self,
            initial_state: Tensor,
    ) -> Tensor:
        assert np.prod(initial_state.shape) == self._n_s
        self._state = initial_state.reshape((self._n_s, 1))
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
    ) -> Tensor:
        return self._A @ self._state + self._B @ action

    def _reward_fn(
            self,
            action: Tensor,
    ) -> Scalar:
        r = self._state.T @ self._Q @ self._state + action.T @ self._R @ action
        return r.item()
