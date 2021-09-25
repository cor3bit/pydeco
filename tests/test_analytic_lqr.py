import pytest

import numpy as np
from scipy.linalg import solve_discrete_are as dare
import jax.numpy as jnp

from pydeco.controller.analytic_lqr import AnalyticalLQR


@pytest.fixture()
def setup():
    # DS params
    A = jnp.array([
        [0.3, 0.7, 0],
        [0.4, 0.5, 0.2],
        [0, 0.2, 0.4],
    ], jnp.float32)
    B = jnp.eye(3, dtype=jnp.float32)

    # reward params
    Q = jnp.eye(3, dtype=jnp.float32)
    R = jnp.eye(3, dtype=jnp.float32)

    return A, B, Q, R


def test_fit(setup):
    A, B, Q, R = setup

    # numpy solution
    np_P = dare(
        np.array(A),
        np.array(B),
        np.array(Q),
        np.array(R),
    )

    # pydeco solution
    lqr = AnalyticalLQR(A, B, Q, R)
    lqr.fit(n_steps=10)

    # comparison
    # P
    np.testing.assert_array_almost_equal(
        np_P,
        np.array(lqr._P),
        decimal=7,
    )

    # K
    np.testing.assert_array_almost_equal(
        np.array([[-0.17794839, -0.39653113, -0.01332207],
                  [-0.2498468, -0.33199662, -0.12628551],
                  [-0.01135366, -0.12226867, -0.21419297]]),
        np.array(lqr._K),
        decimal=7,
    )


def test_simulate_trajectory(setup):
    A, B, Q, R = setup

    # sim params
    x0 = jnp.array([1, 1, 1], jnp.float32)
    t0, tn, n_steps = 0., 1., 20

    lqr = AnalyticalLQR(A, B, Q, R)
    lqr.fit(n_steps=n_steps)
    xs, us, tcost, info = lqr.simulate_trajectory(x0, t0, tn, n_steps)

    assert 1.57543253898 == pytest.approx(tcost, abs=1e-8)
