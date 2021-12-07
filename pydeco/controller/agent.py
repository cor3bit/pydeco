from abc import ABC

from pydeco.types import *
from pydeco.problem.env import Env


class Agent(ABC):
    _cache = None
    _calibrated = False

    def act(
            self,
            state: Tensor,
            **kwargs
    ) -> Tensor:
        raise NotImplementedError

    def train(
            self,
            **kwargs
    ):
        raise NotImplementedError

    def simulate_trajectory(
            self,
            env: Env,
            initial_state: Tensor,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float]:
        raise NotImplementedError
