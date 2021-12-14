from abc import ABC

from pydeco.types import *


class Env(ABC):
    _state = None

    def reset(
            self,
            initial_state: Tensor,
            **kwargs
    ) -> Tensor:
        raise NotImplementedError

    def step(
            self,
            action: Tensor,
            **kwargs
    ) -> Tuple[Scalar, Tensor]:
        # get current reward
        curr_reward = self._reward_fn(action, **kwargs)

        # transition to next state
        next_state = self._transition_fn(action, **kwargs)

        # update internal state
        self._state = next_state

        # return resulting state & reward
        return curr_reward, next_state

    def _transition_fn(
            self,
            action: Tensor,
            **kwargs
    ) -> Tensor:
        raise NotImplementedError

    def _reward_fn(
            self,
            action: Tensor,
            **kwargs
    ) -> Scalar:
        raise NotImplementedError


class MultiAgentEnv(ABC):
    _env_map = None

    def reset(
            self,
            initial_states: Tensors,
            **kwargs
    ) -> Tensors:
        raise NotImplementedError

    def step(
            self,
            agent_id: int,
            action: Tensor,
            **kwargs
    ) -> Sequence[Tuple[Scalar, Tensor]]:
        raise NotImplementedError

    def get_state(self, agent_id: int):
        raise NotImplementedError

    def get_info(self, agent_id: int):
        raise NotImplementedError
