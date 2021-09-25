from abc import ABC, abstractmethod

from pydeco.types import *


class Controller(ABC):
    _cache = None
    _calibrated = False

    def fit(self, **kwargs):
        # find optimal policy
        raise NotImplementedError

    def simulate_trajectory(
            self,
            x0: Tensor,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float, Dict]:
        raise NotImplementedError
