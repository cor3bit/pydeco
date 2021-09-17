from abc import ABC, abstractmethod

from pydeco.types import *


class Controller(ABC):
    _cache = None

    def solve(
            self,
            x0: Tensor,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensors, float, Dict]:
        raise NotImplementedError
