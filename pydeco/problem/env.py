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

    # local cache
    _n_agents = None
    _curr_states = None
    _next_states = None
    _curr_rewards = None
    _curr_actions = None

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

    def roll_states(self):
        self._curr_states = self._next_states
        self._next_states = {i: None for i in range(self._n_agents)}

    def get_state(
            self,
            agent_id: int,
    ) -> Tensor:
        return self._curr_states[agent_id]

    def get_next_state(
            self,
            agent_id: int,
    ) -> Tensor:
        return self._next_states[agent_id]

    def get_action(
            self,
            agent_id: int,
    ) -> Tensor:
        return self._curr_actions[agent_id]

    def get_info(
            self,
            agent_id: int,
    ) -> Tensors:
        neighbors = self._env_map[agent_id].neighbors
        return [self.get_state(neighbor_id) for neighbor_id in neighbors]

    def get_next_info(
            self,
            agent_id: int,
    ) -> Tensors:
        neighbors = self._env_map[agent_id].neighbors
        return [self.get_next_state(neighbor_id) for neighbor_id in neighbors]

    def get_reward(
            self,
            agent_id: int,
    ) -> Scalar:
        return self._curr_rewards[agent_id]

    def set_reward(
            self,
            agent_id: int,
            reward: Scalar,
    ):
        self._curr_rewards[agent_id] = reward

    def set_action(
            self,
            agent_id: int,
            action: Tensor,
    ):
        self._curr_actions[agent_id] = action

    def set_next_state(
            self,
            agent_id: int,
            next_state: Tensor,
    ):
        self._next_states[agent_id] = next_state
