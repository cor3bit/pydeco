import numpy as np

from pydeco.problem.env import Env, MultiAgentEnv
from pydeco.types import *


# model from сaiazzo21
class EVPlatoon(MultiAgentEnv):
    def __init__(
            self,
            n_agents: int,
            communication_map: dict[int, Sequence[int]],
            state_reward_matrix: Tensor,
            action_reward_matrix: Tensor,
    ):
        self._n_agents = n_agents

        # create envs
        # TODO
        self._env_map = {
            i: EVehicle(
                communication_map[i], coupled_dynamics, coupled_rewards,
                system_matrix, control_matrix, state_reward_matrix, action_reward_matrix,
            ) for i in range(n_agents)
        }

    def step(self, agent_id: int, action: Tensor, **kwargs) -> Sequence[Tuple[Scalar, Tensor]]:
        raise NotImplementedError

    def reset(self, initial_states: Tensors, **kwargs) -> Tensors:
        # TODO init individual states + cache
        raise NotImplementedError


class EVehicle(Env):
    def __init__(self, h=0.8):


        self._h = h

    def reset(self, initial_state: Tensor, **kwargs) -> Tensor:
        pass


    def _ev_model(
            self,
            u,
    ):
        p, v = self._state

        part1 = u * nu / R / m

        part2 = g * np.sin(theta)

        part3 = g * np.cos(theta) * Cr (c1 * v + c2) * 1e-3

        part4 = 0.5 * rho * Cd * d *A *v / m

        v_next = part1 - part2 - part3 - part4

        return [v, v_next]

    def _transition_fn(self, action: Tensor, **kwargs) -> Tensor:
        # TODO Euler step
        pass

    def _reward_fn(self, action: Tensor, **kwargs) -> Scalar:
        pass


class DecoupledVehiclePlatoon():
    def __init__(
            self,

            n_vehicles: int,

            # reward params
            state_reward: Tensor = np.ones(3),
            action_reward: Tensor = np.ones(1),

            # logging
            verbose: bool = True,

            # platoon behavior
            leader_velocity: Scalar = 20.,
            vehicle_length: Scalar = 4.5,
            desired_distance: Scalar = 20.,

            # model constants
            mass_of_air: Scalar = 1.293,
            cross_sec_area: Scalar = 2.5,
            drag_coef: Scalar = 0.45,
            mass: Scalar = 1775.,
            engine_const: Scalar = 0.1,
            amplitude: Scalar = 5.,
    ):
        self._N = n_vehicles
        self._n = 3  # state dims: [pos, velocity, acc]
        self._m = 1  # action dims: [engine]

        self._v_ref = leader_velocity
        self._L = vehicle_length
        self._D_ref = desired_distance

        # reward structure
        self._state_reward = state_reward
        self._action_reward = action_reward

        # logging
        self._verbose = verbose

        # model constants
        self._rho = mass_of_air
        self._A_i = cross_sec_area
        self._C_di = drag_coef
        self._mu_i = engine_const
        self._mass = mass
        self._d_mi = amplitude,

        # default states
        self._states = None

    def reset(
            self,
            initial_states: Tensors,
    ):
        assert len(initial_states) == self._N
        self._states = {i: np.array(state_i) for i, state_i in enumerate(initial_states)}

    def step(
            self,
            agent_id: int,
            action: Tensor,
            information: Tensors,
    ) -> Tuple[Tensor, Tensor]:
        curr_state_i = self._states[agent_id]

        # assign reward (coupled)
        curr_reward_i = self._reward_fn(curr_state_i, action, information)

        # propagate state (decoupled)
        next_state_i = self._transition_fn(curr_state_i, action)

        # update internal state
        self._states[agent_id] = next_state_i

        return curr_reward_i, next_state_i

    def _transition_fn(
            self,
            state,
            action,
    ):
        p, v, a = state

        s2 = None
        s3 = None

        return np.array([v, ])

    def _reward_fn(
            self,
            state,
            action,
            information,
    ):
        raise NotImplementedError
