from abc import ABC

from pydeco.types import *


class Env(ABC):
    _state = None

    def reset(
            self,
            initial_state: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def step(
            self,
            action: Tensor,
    ) -> Tuple[Scalar, Tensor]:
        # get current reward
        curr_reward = self._reward_fn(action)

        # transition to next state
        next_state = self._transition_fn(action)

        # update internal state
        self._state = next_state

        # return resulting state & reward
        return curr_reward, next_state

    def _transition_fn(
            self,
            action: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def _reward_fn(
            self,
            action: Tensor,
    ) -> Scalar:
        raise NotImplementedError


class DistributedEnv(ABC):
    def reset(
            self,
            initial_state: Tensors
    ):
        raise NotImplementedError

    def step(
            self,
            agent_id: int,
            action: Tensor,
            information: Tensors,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
