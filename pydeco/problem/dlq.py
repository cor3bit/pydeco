from typing import Optional, List, Union, Tuple

from pydeco.problem.env import Env
from pydeco.types import *


class DLQ(Env):
    def __init__(self, A: Tensor, B: Tensor):
        self._A = A
        self._B = B
        self._x = None
