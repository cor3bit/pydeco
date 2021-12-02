from abc import ABC, abstractmethod
from typing import Callable, Tuple

from pydeco.types import *


class CollaboratingAgent(ABC):
    def act(
            self,
            state: Tensor,
            information: Tensors,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
