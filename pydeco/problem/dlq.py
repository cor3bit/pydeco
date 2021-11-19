from typing import Optional, List, Union, Tuple, Sequence

from pydeco.problem.env import Env
from pydeco.types import *


class DLQ(Env):
    def __init__(
            self,
            # single-agent params
            A: Tensor,
            B: Tensor,
            Q: Tensor,
            R: Tensor,

            # communication
            n_agents: int,
            edges: Sequence[Tuple[int,int]],
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
