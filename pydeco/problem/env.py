from abc import ABC, abstractmethod
from typing import Callable, Tuple

from pydeco.types import *


class Env(ABC):
    def step(
            self,
            u: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
