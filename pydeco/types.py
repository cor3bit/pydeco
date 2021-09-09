from typing import Sequence, Union, Optional, Any, Dict, Tuple, Callable

import jax.numpy as np

Tensor = np.ndarray
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

Scalar = Union[float, Tensor]
Vector = Union[Sequence[float], Tensor]
