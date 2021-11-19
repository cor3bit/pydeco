from typing import Sequence, Union, Optional, Any, Dict, Tuple, Callable

from numpy import ndarray

Tensor = ndarray
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

Scalar = Union[float, Tensor]
Vector = Union[Sequence[float], Tensor]
