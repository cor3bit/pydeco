from abc import ABC, abstractmethod
from typing import Callable, Tuple

from pydeco.types import *


class MultiAgentEnv(ABC):
    def step(
            self,
            agent_id: int,
            action: Tensor,
            information: Tensors,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
