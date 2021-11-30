import numpy as np

from pydeco.problem.ma_env import MultiAgentEnv
from pydeco.types import *


class VehiclePlatoon(MultiAgentEnv):
    def __init__(
            self,
            n_vehicles: int,

            # model constants
            mass_of_air: Scalar = 1.293,
            cross_sec_area: Scalar = 2.5,
            drag_coef: Scalar = 0.45,
            mass: Scalar = 1775.,
            engine_const: Scalar = 0.1,
    ):


        # model constants
        self._rho = mass_of_air
        self._A_i = cross_sec_area
        self._C_di = drag_coef
        self._mu_i = engine_const
        self._mass = mass
        self._d_mi = 5 * n_vehicles, # TODO check that non-vehicle specific



        self._states = dict()

    def step(
            self,
            agent_id: int,
            action: Tensor,
            information: Tensors,
    ) -> Tuple[Tensor, Tensor]:
        curr_state_i = self._states[agent_id]

        # assign reward (coupled)
        curr_reward_i = self._reward_fn(curr_state_i, action)

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
