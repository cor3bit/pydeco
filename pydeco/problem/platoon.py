import numpy as np

from pydeco.problem.env import DistributedEnv
from pydeco.types import *


class VehiclePlatoon(DistributedEnv):
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